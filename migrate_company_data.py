import logging

from sqlalchemy import text
from core.db import get_db
from core.model import Company

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

def migrate_company_symbols():
    """
    Migrates Company symbols from 'AAPL' to 'AAPL.US' (Canonical Format).
    This script updates the Company table and ensures all relationships remain intact.
    """
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        # 1. Identify all companies that need migration
        # We look for symbols that do NOT contain a dot (the non-canonical ones)
        companies_to_migrate = db.query(Company).filter(
            ~Company.symbol.contains('.')
        ).all()

        if not companies_to_migrate:
            logger.info("No companies found requiring migration. Everything is already canonical.")
            return

        logger.info(f"Found {len(companies_to_migrate)} companies to migrate.")

        migration_count = 0
        for company in companies_to_migrate:
            old_symbol = company.symbol
            # Apply your specific logic: e.g., appending '.US'
            # IMPORTANT: Adjust this logic to match your specific provider's requirement
            new_symbol = f"{old_symbol}.US" 
            
            logger.info(f"Migrating: {old_symbol} -> {new_symbol}")

            # 2. Update the Company record
            # Since we are updating the 'symbol' column on the same row, 
            # the primary key (id) remains the same. 
            # All foreign keys in other tables (PriceHistory, etc.) 
            # pointing to this 'id' will remain perfectly valid.
            company.symbol = new_symbol
            
            # 3. If you have any tables that store the SYMBOL as a string (not as a FK),
            # you would update them here. 
            # Example: if some_table.symbol_string == old_symbol: ...
            
            db.add(company)
            migration_count += 1

        # 4. Commit the transaction
        db.commit()
        logger.info(f"Successfully migrated {migration_count} companies.")
        logger.info("Migration complete. All relationships (FKs) are preserved via ID.")

    except Exception as e:
        db.rollback()
        logger.error(f"Migration failed! Rolling back. Error: {e}")
        raise
    finally:
        db.close()

def migrate_company_markets():
    """One-time cleanup script for the market column in the company table."""
    
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        # Strip yfinance '_market' suffixes
        db.execute(text("UPDATE company SET market = LOWER(REPLACE(market, '_market', '')) WHERE market LIKE '%_market';"))
        
        # Map from exchange code
        exchange_to_market = {
            'us': ['NASDAQ', 'NMS', 'NYQ', 'NYSE', 'PCX', 'ASE', 'AMEX', 'OTC'],
            'ca': ['TOR', 'TSX', 'VAN', 'TSXV', 'CSE0', 'NEO'],
            'lon': ['LON', 'LSE'],
            'de': ['GER', 'FRA', 'BER', 'DUS', 'HAM', 'MUN'],
            'jp': ['TYO', 'SPK'],
            'hk': ['HKG'],
            'au': ['ASX']
        }
        
        for market_code, exchanges in exchange_to_market.items():
            db.execute(
                text("UPDATE company SET market = :m WHERE UPPER(exchange) IN :exchs"),
                {"m": market_code, "exchs": tuple(exchanges)}
            )
            
        # Default remaining NULLs to 'us'
        db.execute(text("UPDATE company SET market = 'us' WHERE market IS NULL OR market = '';"))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Migration failed! Rolling back. Error: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    migrate_company_markets()
    