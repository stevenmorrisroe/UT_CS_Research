-- Connect to the database (assuming this is run via psql -d sales_sim_db)
-- \c sales_sim_db; -- Commented out as connection is handled by psql command

-- Drop existing tables IF THEY EXIST
DROP TABLE IF EXISTS sales_records;
DROP TABLE IF EXISTS simulation_logs;

-- Table to log individual messages within each simulation
CREATE TABLE simulation_logs (
    log_id SERIAL PRIMARY KEY,          -- Unique ID for each log entry
    simulation_id UUID NOT NULL,      -- Identifier for a single simulation run
    persona_id VARCHAR(50) NOT NULL,   -- Persona ID used in the simulation (e.g., 'topic_0')
    turn_number INT NOT NULL,           -- Sequence number of the message within the simulation
    role VARCHAR(50) NOT NULL,          -- Role of the sender (Buyer, Seller, Tool, Seller (Tool Call), System)
    content TEXT,                       -- The text content of the message
    tool_name VARCHAR(100),             -- Name of the tool called (if applicable)
    tool_call_id VARCHAR(100),          -- ID associated with the tool call (if applicable)
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP -- Time the message was logged
);

-- Table to record details of completed sales identified during simulations
CREATE TABLE sales_records (
    sale_id SERIAL PRIMARY KEY,         -- Unique ID for each recorded sale
    simulation_id UUID NOT NULL,      -- Foreign key linking to the simulation run
    persona_id VARCHAR(50) NOT NULL,   -- Persona ID associated with the sale
    sold_item_id VARCHAR(255),         -- eBay Item ID of the sold product
    sold_item_name TEXT,              -- Name/Title of the sold product
    sold_item_price NUMERIC(10, 2),     -- Price of the sold item
    product_avg_rank REAL,            -- Avg rank of sold item compared to top K similar items in persona index
    is_verified BOOLEAN DEFAULT FALSE,  -- Flag indicating if a human has verified the sale
    needs_review BOOLEAN DEFAULT TRUE, -- Flag indicating if the sale needs human review (e.g., low confidence)
    sale_confidence REAL,             -- Confidence score (0.0 to 1.0) from the analysis model
    sale_details TEXT,                -- Additional details or justification from the analysis
    sale_timestamp TIMESTAMPTZ NOT NULL -- Timestamp when the sale was detected
);

-- Indexes for faster querying (optional but recommended)
CREATE INDEX idx_simulation_logs_simulation_id ON simulation_logs(simulation_id);
CREATE INDEX idx_sales_records_simulation_id ON sales_records(simulation_id);
CREATE INDEX idx_sales_records_persona_id ON sales_records(persona_id);

-- Grant permissions section removed as it's assumed the user running this owns the objects

-- Print completion message
\echo "Database table and index setup complete." 