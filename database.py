import sqlite3
conn = sqlite3.connect('database.db')
print("Opened database successfully")

conn.execute('CREATE TABLE Account (username TEXT, password TEXT)')
print("Table created successfully")
conn.close()

