import psycopg2
import psycopg2.extras
import pandas as pd
import re
import json
from datetime import datetime
from sqlalchemy import create_engine
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.word_vector import WordVector

DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'dbname': 'seminarSQL',
    'user': 'postgres',
    'password': 'postgres',
}

def get_connection():
    '''สร้างการเชื่อมต่อไปยัง PostgreSQL ด้วยพารามิเตอร์ที่กำหนดใน DB_CONFIG (ใช้กับ psycopg2 โดยตรง เช่น ขั้นตอน schema extraction)'''
    return psycopg2.connect(**DB_CONFIG)

def get_engine():
    '''สร้าง SQLAlchemy engine จาก DB_CONFIG เดียวกัน (ใช้กับ pd.read_sql_query เพื่อไม่ให้ pandas เตือน
    ว่า DBAPI2 connection ธรรมดาไม่ใช่ SQLAlchemy connectable)'''
    url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    return create_engine(url)

def connect():
    try:
        conn = get_connection()
        engine = get_engine()
        print('เชื่อมต่อฐานข้อมูลสำเร็จ')
        return conn, engine
    except Exception as e:
        conn = None
        engine = None
        print(f'เชื่อมต่อฐานข้อมูลไม่สำเร็จ: {e}')

def extract_schema(connection, schema_name='public'):
    '''
    ดึงโครงสร้างฐานข้อมูล (ตาราง, คอลัมน์, ชนิดข้อมูล, primary key, foreign key, comment/alias)
    จาก information_schema + pg_catalog แล้วคืนค่าเป็น dictionary

    'comment' ของแต่ละตาราง/คอลัมน์ (ถ้ามี) มาจาก COMMENT ON TABLE/COLUMN ที่ตั้งไว้ในฐานข้อมูล
    (ดู mock_warehouse_data.sql) ใช้เป็นแหล่งคำพ้อง (alias) ไทย/อังกฤษ แทนการ hardcode
    TABLE_SYNONYMS/COLUMN_SYNONYMS ในโค้ด Python (ขั้นตอนที่ 3 จะนำมาแปลงเป็น dict ต่อ)
    '''
    schema = {}
    if connection is None:
        return schema

    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # ดึงรายชื่อตารางและคอลัมน์ทั้งหมดใน schema
        cur.execute('''
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position;
        ''', (schema_name,))
        for row in cur.fetchall():
            t = row['table_name']
            schema.setdefault(t, {'columns': [], 'primary_key': [], 'foreign_keys': [], 'comment': None})
            schema[t]['columns'].append({
                'name': row['column_name'],
                'data_type': row['data_type'],
                'nullable': row['is_nullable'] == 'YES',
                'comment': None,
            })

        # ดึง primary key ของแต่ละตาราง
        cur.execute('''
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = %s;
        ''', (schema_name,))
        for row in cur.fetchall():
            t = row['table_name']
            if t in schema:
                schema[t]['primary_key'].append(row['column_name'])

        # ดึง foreign key: ตาราง/คอลัมน์ต้นทาง -> ตาราง/คอลัมน์ปลายทาง
        cur.execute('''
            SELECT
                tc.table_name AS source_table,
                kcu.column_name AS source_column,
                ccu.table_name AS target_table,
                ccu.column_name AS target_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
             AND tc.table_schema = ccu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = %s;
        ''', (schema_name,))
        for row in cur.fetchall():
            t = row['source_table']
            if t in schema:
                schema[t]['foreign_keys'].append({
                    'column': row['source_column'],
                    'ref_table': row['target_table'],
                    'ref_column': row['target_column'],
                })

        # ดึง comment ของตาราง (COMMENT ON TABLE ...) ผ่าน pg_catalog เนื่องจาก
        # information_schema ไม่มี comment ให้ใช้โดยตรง
        cur.execute('''
            SELECT c.relname AS table_name, obj_description(c.oid) AS table_comment
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relkind = 'r';
        ''', (schema_name,))
        for row in cur.fetchall():
            t = row['table_name']
            if t in schema:
                schema[t]['comment'] = row['table_comment']

        # ดึง comment ของคอลัมน์ (COMMENT ON COLUMN ...) เช่นเดียวกัน
        cur.execute('''
            SELECT c.relname AS table_name, a.attname AS column_name,
                   col_description(c.oid, a.attnum) AS column_comment
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.oid
            WHERE n.nspname = %s AND c.relkind = 'r'
              AND a.attnum > 0 AND NOT a.attisdropped;
        ''', (schema_name,))
        for row in cur.fetchall():
            t = row['table_name']
            if t in schema:
                for col in schema[t]['columns']:
                    if col['name'] == row['column_name']:
                        col['comment'] = row['column_comment']
                        break
    print(f'ดึงข้อมูล Schema สำเร็จ: {len(schema.keys())} ตาราง -> {', '.join(schema.keys())}')
    print('ได้แก่')
    for tname, clname in schema.items():
        print(f'{tname} -> {', '.join([f"{col['name']} ({col['data_type']})" for col in clname['columns']])
}')
    return schema

def build_table_synonyms(schema):
    '''
    สร้าง dict คำพ้อง (ไทย/อังกฤษ) -> ชื่อตารางจริง จาก schema['comment'] ของแต่ละตาราง
    (มาจาก COMMENT ON TABLE ในฐานข้อมูล คั่นแต่ละคำด้วย '|') โดยเติมชื่อตารางจริงเป็นคำพ้องให้อัตโนมัติเสมอ
    '''
    synonyms = {}
    for table_name, info in schema.items():
        aliases = [table_name]
        comment = info.get('comment')
        if comment:
            aliases += [a.strip() for a in comment.split('|') if a.strip()]
        for alias in aliases:
            synonyms[alias.lower()] = table_name
    return synonyms


def build_column_synonyms(schema):
    '''
    สร้าง dict คำพ้อง (ไทย/อังกฤษ) -> ชื่อคอลัมน์จริง จาก column['comment'] ของทุกตาราง
    (มาจาก COMMENT ON COLUMN ในฐานข้อมูล คั่นแต่ละคำด้วย '|') ใช้กับ intent แบบ SUM เป็นหลัก
    ถ้าหลายตารางมีคอลัมน์ชื่อเดียวกัน (เช่น quantity) และมี comment ต่างกัน จะรวมคำพ้องจากทุกที่เข้าด้วยกัน
    '''
    synonyms = {}
    for info in schema.values():
        for col in info['columns']:
            aliases = [col['name']]
            comment = col.get('comment')
            if comment:
                aliases += [a.strip() for a in comment.split('|') if a.strip()]
            for alias in aliases:
                synonyms[alias.lower()] = col['name']
    return synonyms

def detect_intent(text):
    '''ตรวจจับ intent (COUNT, SUM, SELECT) จากคำสำคัญในประโยคคำถาม'''
    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return intent
    return 'SELECT'  # ค่าเริ่มต้นถ้าไม่พบ intent ที่ชัดเจน


def detect_table(text):
    '''ตรวจจับตาราง/entity เป้าหมายของคำถามจาก TABLE_SYNONYMS โดยเลือกคำที่ปรากฏก่อนสุดในประโยค
    (กันปัญหาประโยคที่มีคำพ้องหลายคำ เช่น "สินค้าของผู้จำหน่าย" ต้องเลือก products ไม่ใช่ suppliers)'''
    text_lower = text.lower()
    best_pos, best_table = None, None
    for keyword, table in TABLE_SYNONYMS.items():
        pos = text_lower.find(keyword)
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos, best_table = pos, table
    return best_table


def detect_aggregate_column(text):
    '''ตรวจจับคอลัมน์ที่ต้องการคำนวณ (ใช้กับ intent แบบ SUM) จาก COLUMN_SYNONYMS'''
    text_lower = text.lower()
    for keyword, column in COLUMN_SYNONYMS.items():
        if keyword in text_lower:
            return column
    return None


def detect_name_filter(text):
    '''ตรวจจับเงื่อนไขกรองด้วยชื่อที่ระบุในประโยค เช่น "ชื่อสมชาย", "ผู้จำหน่ายชื่อไทยพาร์ทส์" หรือ "named John"
    คืนค่าเป็น dict {'value': ค่าที่ค้นหา, 'entity_table': ตารางที่คำว่า "ชื่อ" อ้างถึง (ถ้าระบุชัดเจน)}
    หรือ None ถ้าไม่พบ

    entity_table ใช้คำพ้อง (TABLE_SYNONYMS) ที่อยู่ *ก่อนหน้า* คำว่า "ชื่อ" ทันที เพื่อรู้ว่าต้องกรองชื่อ
    ของตารางไหน (เช่น "ผู้จำหน่ายชื่อไทยพาร์ทส์" -> suppliers) ถ้าไม่พบคำระบุ entity ชัดเจน จะคืนค่า
    entity_table เป็น None แล้วปล่อยให้ generate_sql ใช้ตารางหลักของคำถามแทน
    '''
    patterns = [
        r'ชื่อ\s*([ก-๙a-zA-Z0-9]+)',
        r'named\s+([a-zA-Z]+)',
        r'name\s+([a-zA-Z]+)',
    ]
    value, match_start = None, None
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            value, match_start = m.group(1), m.start()
            break
    if value is None:
        return None

    text_lower = text.lower()
    entity_table, best_pos = None, -1
    for keyword, table in TABLE_SYNONYMS.items():
        pos = text_lower.rfind(keyword, 0, match_start)
        if pos != -1 and pos > best_pos:
            best_pos, entity_table = pos, table

    return {'value': value, 'entity_table': entity_table}


def detect_date_filter(text):
    '''ตรวจจับเงื่อนไขวันที่แบบง่าย เช่น "เดือนนี้" หรือ "this month"'''
    text_lower = text.lower()
    today = datetime.now()
    if 'เดือนนี้' in text_lower or 'this month' in text_lower:
        start = today.replace(day=1).strftime('%Y-%m-%d')
        return {'type': 'month', 'start': start}
    return None


def extract_intent_entities(text):
    '''
    ฟังก์ชันหลักของขั้นตอนที่ 4: รวมผลจากฟังก์ชันตรวจจับด้านบนทั้งหมด
    คืนค่าเป็น dict ที่มี intent, table, aggregate_column, filters
    เพื่อนำไปสร้าง SQL ในขั้นตอนที่ 5 ต่อไป
    '''
    intent = detect_intent(text)
    table = detect_table(text)
    agg_col = detect_aggregate_column(text) if intent == 'SUM' else None
    name_filter = detect_name_filter(text)
    date_filter = detect_date_filter(text)

    filters = {}
    if name_filter:
        filters['name_filter'] = name_filter
    if date_filter:
        filters['date'] = date_filter

    return {
        'raw_text': text,
        'intent': intent,
        'table': table,
        'aggregate_column': agg_col,
        'filters': filters,
    }

def find_join_path(schema, table_a, table_b):
    '''หา foreign key ที่เชื่อม table_a กับ table_b โดยตรง คืนค่าเป็น join condition (string) หรือ None ถ้าไม่พบ'''
    if table_a not in schema or table_b not in schema:
        return None
    for fk in schema[table_a].get('foreign_keys', []):
        if fk['ref_table'] == table_b:
            return f"{table_a}.{fk['column']} = {table_b}.{fk['ref_column']}"
    for fk in schema[table_b].get('foreign_keys', []):
        if fk['ref_table'] == table_a:
            return f"{table_b}.{fk['column']} = {table_a}.{fk['ref_column']}"
    return None


# ชื่อคอลัมน์วันที่/เวลาที่พบได้ในตารางต่างๆ ของฐานข้อมูล เรียงเป็นลำดับการค้นหา
DATE_COLUMN_CANDIDATES = ['order_date', 'movement_date', 'last_updated', 'created_at']


def generate_sql(parsed, schema):
    '''
    ขั้นตอนที่ 5: แปลงผลลัพธ์จาก extract_intent_entities ให้เป็น SQL statement จริง
    โดยอ้างอิงชื่อตาราง/คอลัมน์จาก schema ที่ดึงมาได้จริง (ขั้นตอนที่ 2) และ join ตาราง
    อัตโนมัติถ้าต้อง filter ด้วยคอลัมน์ที่อยู่คนละตารางกับตารางหลัก
    คืนค่าเป็น (sql, params, error) โดยใช้ %s placeholder เพื่อป้องกัน SQL injection
    '''
    table = parsed['table']
    intent = parsed['intent']
    filters = parsed['filters']
    agg_col_requested = parsed['aggregate_column']
    params = []

    # ถ้าไม่พบตารางตรงๆ แต่เป็น intent SUM ที่ระบุคอลัมน์ไว้ (เช่น "ราคาเฉลี่ยเท่าไหร่"
    # ไม่มีคำว่าตารางเลย) ให้เดาตารางจากคอลัมน์นั้นโดยค้นหาใน schema ว่าตารางไหนมีคอลัมน์นี้
    if (table is None or table not in schema) and intent == 'SUM' and agg_col_requested:
        for t, info in schema.items():
            if agg_col_requested in [c['name'] for c in info['columns']]:
                table = t
                break

    if table is None or table not in schema:
        return None, params, 'ไม่พบตารางที่เกี่ยวข้องกับคำถามนี้ในฐานข้อมูล'

    joins = []
    where_clauses = []

    # ถ้ามีการระบุชื่อ (เช่น "ผู้จำหน่ายชื่อไทยพาร์ทส์") ให้หาว่าต้องกรองด้วยคอลัมน์ name ของตารางไหน
    # แล้ว join ไปตารางนั้นอัตโนมัติถ้าจำเป็น (ตารางเป้าหมายต้องมีคอลัมน์ name จริงในฐานข้อมูล)
    if 'name_filter' in filters:
        name_val = filters['name_filter']['value']
        entity_table = filters['name_filter']['entity_table']
        # ถ้าไม่ได้ระบุ entity ชัดเจน (เช่นพิมพ์ลอยๆ ว่า "ชื่อสมชาย") ให้ถือว่าหมายถึงตารางหลักของคำถาม
        target_table = entity_table if (entity_table and entity_table in schema) else table

        target_cols = [c['name'] for c in schema[target_table]['columns']]
        if 'name' not in target_cols:
            return None, params, f"ตาราง {target_table} ไม่มีคอลัมน์ 'name' ให้กรองด้วยชื่อได้"

        if target_table == table:
            where_clauses.append(f"{table}.name ILIKE %s")
            params.append(f"%{name_val}%")
        else:
            join_condition = find_join_path(schema, table, target_table)
            if join_condition:
                joins.append(f"JOIN {target_table} ON {join_condition}")
                where_clauses.append(f"{target_table}.name ILIKE %s")
                params.append(f"%{name_val}%")
            else:
                return None, params, f"ไม่พบความสัมพันธ์ (foreign key) ระหว่างตาราง {table} กับ {target_table}"

    # ถ้ามีเงื่อนไขวันที่ และตารางมีคอลัมน์วันที่/เวลาที่รู้จัก ให้เพิ่มเงื่อนไข
    if 'date' in filters:
        date_info = filters['date']
        col_names = [c['name'] for c in schema[table]['columns']]
        date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in col_names), None)
        if date_col:
            where_clauses.append(f"{table}.{date_col} >= %s")
            params.append(date_info['start'])

    # กำหนด SELECT clause ตาม intent ที่ตรวจพบ
    if intent == 'COUNT':
        select_clause = f"SELECT COUNT(*) AS total FROM {table}"
    elif intent == 'SUM':
        col_names = [c['name'] for c in schema[table]['columns']]
        agg_col = agg_col_requested or ('quantity' if 'quantity' in col_names else None)
        if agg_col is None or agg_col not in col_names:
            return None, params, f"ไม่พบคอลัมน์ '{agg_col}' ในตาราง {table}"
        select_clause = f"SELECT SUM({table}.{agg_col}) AS total FROM {table}"
    else:
        select_clause = f"SELECT * FROM {table}"

    sql = select_clause
    if joins:
        sql += ' ' + ' '.join(joins)
    if where_clauses:
        sql += ' WHERE ' + ' AND '.join(where_clauses)
    sql += ';'
    return sql, params, None

def run_nlp_query(text, connection, schema, sql_engine=None):
    '''
    ขั้นตอนที่ 6: รับคำถามภาษาธรรมชาติ 1 ประโยค แล้วรันผ่านทั้ง pipeline
    (extract_intent_entities -> generate_sql -> execute) และแสดงผลเป็น DataFrame
    มีการดักจับ error แบบง่ายๆ เพื่อไม่ให้ notebook หยุดทำงานกลางคัน

    sql_engine: SQLAlchemy engine (ถ้ามี) ใช้กับ pd.read_sql_query แทน psycopg2 connection ตรงๆ
    เพื่อไม่ให้ pandas เตือนเรื่อง DBAPI2 connection ที่ไม่ใช่ SQLAlchemy connectable
    '''
    parsed = extract_intent_entities(text)
    sql, params, error = generate_sql(parsed, schema)

    print(f"คำถาม: {text}")
    print(f"Intent ที่ตรวจพบ: {parsed['intent']} | ตารางที่ตรวจพบ: {parsed['table']}")

    if error:
        print(f"ไม่สามารถสร้าง SQL ได้: {error}")
        return None

    print(f"SQL ที่ generate:\n{sql}")
    if params:
        print(f"พารามิเตอร์: {params}")

    read_target = sql_engine if sql_engine is not None else connection
    if read_target is None:
        print('ไม่มีการเชื่อมต่อฐานข้อมูลจริง จึงไม่สามารถรัน query ได้ (แสดงเฉพาะ SQL ที่ generate ด้านบน)')
        return None

    try:
        # SQLAlchemy engine ตีความ list ว่าเป็นหลายชุดพารามิเตอร์ (executemany) จึงต้องแปลงเป็น tuple ก่อนเสมอ
        df = pd.read_sql_query(sql, read_target, params=tuple(params))
        print(f"ผลลัพธ์: พบ {len(df)} แถว")
        return df
    except Exception as e:
        print(f"เกิดข้อผิดพลาดขณะรัน query: {e}")
        return None
    
if __name__ == '__main__':
    conn, engine = connect()
    DB_SCHEMA = extract_schema(conn)
    
    TABLE_SYNONYMS = build_table_synonyms(DB_SCHEMA)
    COLUMN_SYNONYMS = build_column_synonyms(DB_SCHEMA)
    INTENT_KEYWORDS = {
        'COUNT': ['กี่คน', 'กี่ชิ้น', 'กี่รายการ', 'กี่แห่ง', 'จำนวนทั้งหมด', 'count', 'how many'],
        'SUM': ['รวมยอด', 'ยอดรวม', 'ทั้งหมดกี่', 'เท่าไหร่', 'sum', 'total'],
        'SELECT': ['แสดง', 'ดู', 'รายการ', 'show', 'list', 'display'],
    }
    
    demo_questions = [
        'มีสินค้ากี่รายการ',
        'คลังสินค้ามีกี่แห่ง',
        'จำนวนคงเหลือของสินค้าทั้งหมดเท่าไหร่',
        'แสดงสินค้าของผู้จำหน่ายชื่อไทยพาร์ทส์',
        'แสดงคลังสินค้าทั้งหมด',
    ]

    for q in demo_questions:
        print('=' * 70)
        result_df = run_nlp_query(q, conn, DB_SCHEMA, sql_engine=engine)
        if result_df is not None:
            print(result_df)
        print()
    
    