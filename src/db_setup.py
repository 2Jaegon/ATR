import os
import sqlite3

DB_PATH = "factory_logs.db"

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def setup_database():
    print(f"Setting up local SQLite DB ({DB_PATH})...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. 유지보수 이력 테이블 생성
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS maintenance_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        pattern_desc TEXT,
        
        -- 17 Core Sensors
        IONGAUGEPRESSURE REAL,
        ETCHBEAMVOLTAGE REAL,
        ETCHBEAMCURRENT REAL,
        ETCHSUPPRESSORVOLTAGE REAL,
        ETCHSUPPRESSORCURRENT REAL,
        FLOWCOOLFLOWRATE REAL,
        FLOWCOOLPRESSURE REAL,
        ETCHGASCHANNEL1READBACK REAL,
        ETCHPBNGASREADBACK REAL,
        FIXTURETILTANGLE REAL,
        ROTATIONSPEED REAL,
        ACTUALROTATIONANGLE REAL,
        FIXTURESHUTTERPOSITION REAL,
        ETCHSOURCEUSAGE REAL,
        ETCHAUXSOURCETIMER REAL,
        ETCHAUX2SOURCETIMER REAL,
        ACTUALSTEPDURATION REAL,
        
        cause TEXT,
        action_taken TEXT,
        success_rate REAL,
        is_false_alarm INTEGER DEFAULT 0
    )
    """
    
    cursor.execute(create_table_sql)
    
    # 2. 샘플 데이터 주입 (테이블이 비어있을 경우만)
    cursor.execute("SELECT COUNT(*) FROM maintenance_history")
    count = cursor.fetchone()[0]
    
    if count == 0:
        insert_sql = """
        INSERT INTO maintenance_history (
            pattern_desc, IONGAUGEPRESSURE, ETCHBEAMVOLTAGE, ETCHBEAMCURRENT, 
            ETCHSUPPRESSORVOLTAGE, ETCHSUPPRESSORCURRENT, FLOWCOOLFLOWRATE, FLOWCOOLPRESSURE, 
            ETCHGASCHANNEL1READBACK, ETCHPBNGASREADBACK, FIXTURETILTANGLE, ROTATIONSPEED, 
            ACTUALROTATIONANGLE, FIXTURESHUTTERPOSITION, ETCHSOURCEUSAGE, ETCHAUXSOURCETIMER, 
            ETCHAUX2SOURCETIMER, ACTUALSTEPDURATION, 
            cause, action_taken, success_rate, is_false_alarm
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
        """
        sample_data = [
            ("Flowcool Pressure Too High", 0.1, 0.2, -0.1, 0.0, 0.0, 2.5, 3.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Pump malfunction", "Replace Flowcool Pump", 0.95, 0),
            ("Flowcool leak", 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Pipe leakage", "Seal pipe joints", 0.88, 0),
            ("Unknown Sensor Glitch", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Sensor Glitch", "Ignore", 0.99, 1)
        ]
        cursor.executemany(insert_sql, sample_data)
        conn.commit()
        print("Table created and sample data inserted.")
    else:
        print("Database already initialized.")
        
    conn.close()

if __name__ == "__main__":
    setup_database()

