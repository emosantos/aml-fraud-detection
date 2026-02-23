# scripts/test_connection.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from config.config import DB_CONFIG

def test_database_connection():
    """Test PostgreSQL connection"""
    
    print("="*70)
    print("DATABASE CONNECTION TEST")
    print("="*70)
    
    # Show what we're trying to connect to
    print(f"\n🔍 Connection details:")
    print(f"   Host:     {DB_CONFIG['host']}")
    print(f"   Port:     {DB_CONFIG['port']}")
    print(f"   Database: {DB_CONFIG['database']}")
    print(f"   User:     {DB_CONFIG['user']}")
    print(f"   Password: {'*' * len(DB_CONFIG['password'])} (hidden)")
    
    print(f"\n🔌 Attempting to connect...\n")
    
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        
        print("✅ Connection established!")
        
        cursor = conn.cursor()
        
        # Test 1: Database version
        print("\n📊 Running tests...")
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"✅ PostgreSQL version: {db_version[0][:50]}...")
        
        # Test 2: Check schemas exist
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name IN ('bronze_layer', 'silver_layer', 'gold_layer')
            ORDER BY schema_name
        """)
        schemas = cursor.fetchall()
        
        if schemas:
            schema_names = [s[0] for s in schemas]
            print(f"✅ Found schemas: {schema_names}")
        else:
            print("⚠️  Warning: No custom schemas found (bronze_layer, silver_layer, gold_layer)")
            print("   This is normal if you haven't run schema.sql yet")
        
        # Test 3: List all schemas
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT LIKE 'pg_%' 
            AND schema_name != 'information_schema'
            ORDER BY schema_name
        """)
        all_schemas = cursor.fetchall()
        print(f"✅ All schemas in database: {[s[0] for s in all_schemas]}")
        
        cursor.close()
        conn.close()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        
    except psycopg2.OperationalError as e:
        print("❌ Connection failed!")
        print(f"\n🔴 Error type: OperationalError")
        print(f"🔴 Error message: {e}")
        print("\n📋 Troubleshooting steps:")
        print("1. Check if Docker Desktop is running")
        print("2. Check if containers are up: docker-compose ps")
        print("3. Start containers: docker-compose up -d")
        print("4. Check container logs: docker-compose logs postgres")
        print("5. Verify .env file has correct credentials")
        
    except psycopg2.Error as e:
        print("❌ Database error!")
        print(f"\n🔴 Error: {e}")
        
    except Exception as e:
        print("❌ Unexpected error!")
        print(f"\n🔴 Error type: {type(e).__name__}")
        print(f"🔴 Error message: {e}")
        import traceback
        print("\n🔍 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_database_connection()