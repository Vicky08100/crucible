//! Contract event indexing service.
//!
//! Indexes Soroban contract events from the Stellar network into a queryable
//! local store backed by PostgreSQL, with Redis caching for hot queries.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

/// A single indexed contract event.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct IndexedEvent {
    pub id: Uuid,
    pub contract_id: String,
    pub ledger_sequence: i64,
    pub transaction_hash: String,
    pub event_index: i32,
    pub event_type: String,
    pub topics: serde_json::Value,
    pub data: serde_json::Value,
    pub indexed_at: DateTime<Utc>,
}

/// Query parameters for filtering indexed events.
#[derive(Debug, Clone, Default)]
pub struct EventQuery {
    pub contract_id: Option<String>,
    pub event_type: Option<String>,
    pub from_ledger: Option<i64>,
    pub to_ledger: Option<i64>,
    pub limit: Option<i64>,
}

/// Service responsible for ingesting and querying contract events.
pub struct EventIndexer {
    db: PgPool,
}

impl EventIndexer {
    pub fn new(db: PgPool) -> Self {
        Self { db }
    }

    /// Ingest a batch of raw events from the Stellar network.
    #[instrument(skip(self, events), fields(count = events.len()))]
    pub async fn ingest(&self, events: Vec<RawEvent>) -> Result<usize, sqlx::Error> {
        if events.is_empty() {
            return Ok(0);
        }

        let mut inserted = 0;
        for event in events {
            let result = sqlx::query(
                r#"
                INSERT INTO contract_events
                    (id, contract_id, ledger_sequence, transaction_hash, event_index, event_type, topics, data, indexed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (transaction_hash, event_index) DO NOTHING
                "#,
            )
            .bind(Uuid::new_v4())
            .bind(&event.contract_id)
            .bind(event.ledger_sequence)
            .bind(&event.transaction_hash)
            .bind(event.event_index)
            .bind(&event.event_type)
            .bind(&event.topics)
            .bind(&event.data)
            .bind(Utc::now())
            .execute(&self.db)
            .await?;

            if result.rows_affected() > 0 {
                inserted += 1;
                debug!(contract_id = %event.contract_id, event_type = %event.event_type, "Indexed event");
            }
        }

        info!(inserted, "Event ingestion complete");
        Ok(inserted)
    }

    /// Query indexed events with optional filters.
    #[instrument(skip(self))]
    pub async fn query(&self, q: EventQuery) -> Result<Vec<IndexedEvent>, sqlx::Error> {
        // Build a dynamic query using a base + optional WHERE clauses.
        // sqlx doesn't support fully dynamic queries, so we use a fixed
        // parameterised form that covers all filter combinations.
        let limit = q.limit.unwrap_or(100).min(1000);

        let rows = sqlx::query_as::<_, IndexedEvent>(
            r#"
                 SELECT id, contract_id, ledger_sequence, transaction_hash, event_index,
                   event_type, topics, data, indexed_at
            FROM contract_events
            WHERE ($1::text IS NULL OR contract_id = $1)
              AND ($2::text IS NULL OR event_type  = $2)
              AND ($3::bigint IS NULL OR ledger_sequence >= $3)
              AND ($4::bigint IS NULL OR ledger_sequence <= $4)
            ORDER BY ledger_sequence DESC, indexed_at DESC
            LIMIT $5
            "#,
        )
        .bind(q.contract_id)
        .bind(q.event_type)
        .bind(q.from_ledger)
        .bind(q.to_ledger)
        .bind(limit)
        .fetch_all(&self.db)
        .await?;

        Ok(rows)
    }

    /// Return the highest ledger sequence that has been indexed.
    pub async fn latest_ledger(&self) -> Result<Option<i64>, sqlx::Error> {
        let row: Option<(Option<i64>,)> =
            sqlx::query_as("SELECT MAX(ledger_sequence) FROM contract_events")
                .fetch_optional(&self.db)
                .await?;
        Ok(row.and_then(|(v,)| v))
    }

    /// Delete events older than `keep_ledgers` ledgers behind the latest.
    #[instrument(skip(self))]
    pub async fn prune(&self, keep_ledgers: i64) -> Result<u64, sqlx::Error> {
        let latest = self.latest_ledger().await?.unwrap_or(0);
        let cutoff = latest.saturating_sub(keep_ledgers);
        let result = sqlx::query("DELETE FROM contract_events WHERE ledger_sequence < $1")
            .bind(cutoff)
            .execute(&self.db)
            .await?;
        let deleted = result.rows_affected();
        if deleted > 0 {
            warn!(deleted, cutoff, "Pruned old contract events");
        }
        Ok(deleted)
    }

    /// Reconciles a potential blockchain reorganization when receiving an incoming ledger header.
    /// If the incoming ledger's parent hash does not match the recorded previous block hash,
    /// a reorganization is detected, the common ancestor is located, and orphaned events
    /// are purged in a single atomic database transaction.
    #[instrument(skip(self))]
    pub async fn reconcile_reorg(
        &self,
        new_ledger: &LedgerBlockHeader,
        ancestor_candidates: Option<&[LedgerBlockHeader]>,
    ) -> Result<Option<ReorgReconciliation>, sqlx::Error> {
        let latest = self.latest_ledger_header().await?;
        let Some(prev) = latest else {
            self.record_ledger_header(new_ledger).await?;
            return Ok(None);
        };

        // Continuity check: contiguous chain requires new sequence == prev + 1 and new parent_hash == prev block_hash
        let is_contiguous = new_ledger.sequence == prev.sequence + 1
            && new_ledger.parent_hash == prev.block_hash;

        if is_contiguous {
            self.record_ledger_header(new_ledger).await?;
            return Ok(None);
        }

        warn!(
            prev_seq = prev.sequence,
            prev_hash = %prev.block_hash,
            new_seq = new_ledger.sequence,
            new_parent_hash = %new_ledger.parent_hash,
            "Blockchain reorganization detected: parent hash mismatch"
        );

        let common_ancestor = self
            .locate_common_ancestor(&prev, new_ledger, ancestor_candidates)
            .await?;

        // Atomically rollback orphaned events and headers beyond common ancestor
        let mut tx = self.db.begin().await?;

        let purge_result = sqlx::query("DELETE FROM contract_events WHERE ledger_sequence > $1")
            .bind(common_ancestor)
            .execute(&mut *tx)
            .await?;
        let purged_events_count = purge_result.rows_affected();

        let _ = sqlx::query("DELETE FROM indexed_ledger_headers WHERE sequence > $1")
            .bind(common_ancestor)
            .execute(&mut *tx)
            .await;

        let _ = sqlx::query(
            r#"
            INSERT INTO indexed_ledger_headers (sequence, block_hash, parent_hash, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (sequence) DO UPDATE
            SET block_hash = EXCLUDED.block_hash,
                parent_hash = EXCLUDED.parent_hash,
                created_at = NOW()
            "#,
        )
        .bind(new_ledger.sequence)
        .bind(&new_ledger.block_hash)
        .bind(&new_ledger.parent_hash)
        .execute(&mut *tx)
        .await;

        tx.commit().await?;

        info!(
            common_ancestor,
            orphaned_from = common_ancestor + 1,
            purged_events_count,
            "Reorganization reconciliation completed"
        );

        Ok(Some(ReorgReconciliation {
            reorg_detected: true,
            orphaned_from_ledger: common_ancestor + 1,
            common_ancestor_ledger: common_ancestor,
            purged_events_count,
        }))
    }

    /// Rollback orphaned event records above common_ancestor_sequence in an atomic database transaction.
    #[instrument(skip(self))]
    pub async fn rollback_orphaned_events(&self, common_ancestor_sequence: i64) -> Result<u64, sqlx::Error> {
        let mut tx = self.db.begin().await?;

        let result = sqlx::query("DELETE FROM contract_events WHERE ledger_sequence > $1")
            .bind(common_ancestor_sequence)
            .execute(&mut *tx)
            .await?;
        let deleted = result.rows_affected();

        let _ = sqlx::query("DELETE FROM indexed_ledger_headers WHERE sequence > $1")
            .bind(common_ancestor_sequence)
            .execute(&mut *tx)
            .await;

        tx.commit().await?;

        if deleted > 0 {
            warn!(deleted, common_ancestor_sequence, "Rolled back orphaned contract events");
        }
        Ok(deleted)
    }

    /// Locates the common ancestor ledger sequence between the indexed history and incoming chain.
    pub async fn locate_common_ancestor(
        &self,
        prev: &LedgerBlockHeader,
        new_ledger: &LedgerBlockHeader,
        ancestor_candidates: Option<&[LedgerBlockHeader]>,
    ) -> Result<i64, sqlx::Error> {
        // Direct parent match in indexed headers
        let parent_match: Option<(i64,)> = sqlx::query_as(
            "SELECT sequence FROM indexed_ledger_headers WHERE block_hash = $1",
        )
        .bind(&new_ledger.parent_hash)
        .fetch_optional(&self.db)
        .await
        .unwrap_or(None);

        if let Some((seq,)) = parent_match {
            return Ok(seq);
        }

        // Search through candidate headers from incoming branch
        if let Some(candidates) = ancestor_candidates {
            for candidate in candidates.iter().rev() {
                let match_exists: Option<(i64,)> = sqlx::query_as(
                    "SELECT sequence FROM indexed_ledger_headers WHERE sequence = $1 AND block_hash = $2",
                )
                .bind(candidate.sequence)
                .bind(&candidate.block_hash)
                .fetch_optional(&self.db)
                .await
                .unwrap_or(None);

                if let Some((seq,)) = match_exists {
                    return Ok(seq);
                }
            }
        }

        // Conservative fallback: rollback boundary at least before new ledger or previous sequence
        Ok((new_ledger.sequence - 1).min(prev.sequence).max(0))
    }

    /// Fetches the latest indexed ledger header.
    pub async fn latest_ledger_header(&self) -> Result<Option<LedgerBlockHeader>, sqlx::Error> {
        let row = sqlx::query_as::<_, LedgerBlockHeader>(
            "SELECT sequence, block_hash, parent_hash FROM indexed_ledger_headers ORDER BY sequence DESC LIMIT 1",
        )
        .fetch_optional(&self.db)
        .await;

        match row {
            Ok(header) => Ok(header),
            Err(_) => {
                let _ = self.ensure_header_table().await;
                Ok(None)
            }
        }
    }

    /// Records or updates a ledger header.
    pub async fn record_ledger_header(&self, header: &LedgerBlockHeader) -> Result<(), sqlx::Error> {
        let _ = self.ensure_header_table().await;
        sqlx::query(
            r#"
            INSERT INTO indexed_ledger_headers (sequence, block_hash, parent_hash, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (sequence) DO UPDATE
            SET block_hash = EXCLUDED.block_hash,
                parent_hash = EXCLUDED.parent_hash,
                created_at = NOW()
            "#,
        )
        .bind(header.sequence)
        .bind(&header.block_hash)
        .bind(&header.parent_hash)
        .execute(&self.db)
        .await?;
        Ok(())
    }

    /// Ensure the ledger headers table exists.
    async fn ensure_header_table(&self) -> Result<(), sqlx::Error> {
        let _ = sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS indexed_ledger_headers (
                sequence BIGINT PRIMARY KEY,
                block_hash TEXT NOT NULL,
                parent_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_indexed_ledger_headers_hash
                ON indexed_ledger_headers (block_hash);
            "#,
        )
        .execute(&self.db)
        .await;
        Ok(())
    }
}

/// Ledger block header used to track continuity and detect chain reorganizations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, sqlx::FromRow)]
pub struct LedgerBlockHeader {
    pub sequence: i64,
    pub block_hash: String,
    pub parent_hash: String,
}

/// Result of reorganization detection and reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReorgReconciliation {
    pub reorg_detected: bool,
    pub orphaned_from_ledger: i64,
    pub common_ancestor_ledger: i64,
    pub purged_events_count: u64,
}

/// A raw event as received from the Stellar network / horizon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEvent {
    pub contract_id: String,
    pub ledger_sequence: i64,
    pub transaction_hash: String,
    pub event_index: i32,
    pub event_type: String,
    pub topics: serde_json::Value,
    pub data: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::postgres::PgPoolOptions;

    fn lazy_pool() -> PgPool {
        PgPoolOptions::new()
            .max_connections(1)
            .connect_lazy("postgres://postgres:postgres@localhost/crucible_test")
            .unwrap()
    }

    fn make_event(contract_id: &str, ledger: i64, tx: &str, kind: &str) -> RawEvent {
        RawEvent {
            contract_id: contract_id.to_string(),
            ledger_sequence: ledger,
            transaction_hash: tx.to_string(),
            event_index: 0,
            event_type: kind.to_string(),
            topics: serde_json::json!(["transfer"]),
            data: serde_json::json!({"amount": 100}),
        }
    }

    #[tokio::test]
    async fn test_ingest_empty_batch_returns_zero() {
        let indexer = EventIndexer::new(lazy_pool());
        let result = indexer.ingest(vec![]).await.unwrap();
        assert_eq!(result, 0);
    }

    #[tokio::test]
    async fn test_event_query_defaults() {
        let q = EventQuery::default();
        assert!(q.contract_id.is_none());
        assert!(q.limit.is_none());
    }

    #[test]
    fn test_raw_event_serialization() {
        let ev = make_event("CABC", 100, "txhash1", "transfer");
        let json = serde_json::to_string(&ev).unwrap();
        let back: RawEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.contract_id, "CABC");
        assert_eq!(back.ledger_sequence, 100);
        assert_eq!(back.event_index, 0);
    }

    #[test]
    fn test_ledger_block_header_serialization() {
        let header = LedgerBlockHeader {
            sequence: 500,
            block_hash: "hash_500".to_string(),
            parent_hash: "hash_499".to_string(),
        };
        let json = serde_json::to_string(&header).unwrap();
        let back: LedgerBlockHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sequence, 500);
        assert_eq!(back.block_hash, "hash_500");
        assert_eq!(back.parent_hash, "hash_499");
    }

    #[test]
    fn test_reorg_reconciliation_result() {
        let rec = ReorgReconciliation {
            reorg_detected: true,
            orphaned_from_ledger: 105,
            common_ancestor_ledger: 104,
            purged_events_count: 12,
        };
        assert!(rec.reorg_detected);
        assert_eq!(rec.orphaned_from_ledger, 105);
        assert_eq!(rec.common_ancestor_ledger, 104);
        assert_eq!(rec.purged_events_count, 12);
    }
}
