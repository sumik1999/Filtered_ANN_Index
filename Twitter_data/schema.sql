-- Logical schema for temporal pre-filtering. Generated data will live in a
-- local DuckDB database under Twitter_data/sql/.
CREATE TABLE tweets (
    vector_id INTEGER PRIMARY KEY,
    tweet_id VARCHAR NOT NULL UNIQUE,
    timestamp_ms BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    language VARCHAR,
    is_retweet BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX idx_tweets_timestamp_ms ON tweets(timestamp_ms);

-- Single-range eligibility query (half-open interval):
-- SELECT vector_id
-- FROM tweets
-- WHERE timestamp_ms >= ? AND timestamp_ms < ?
-- ORDER BY vector_id;
--
-- A union of ranges is represented by additional parameterized OR clauses.
