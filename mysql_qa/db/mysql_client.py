"""MySQL access for RetailRAG FAQ and conversation data."""

from pathlib import Path

import pandas as pd
import pymysql

from base import Config, logger


class MySQLClient:
    def __init__(self):
        self.logger = logger
        config = Config()
        try:
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                port=config.MYSQL_PORT,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE,
                connect_timeout=5,
                read_timeout=30,
                write_timeout=30,
            )
            self.cursor = self.connection.cursor()
            self.logger.info("MySQL 连接成功")
        except pymysql.MySQLError as exc:
            self.logger.error("MySQL 连接失败: %s", exc)
            raise

    def create_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jpkb (
                id INT AUTO_INCREMENT PRIMARY KEY,
                category VARCHAR(50) NOT NULL,
                question VARCHAR(1000) NOT NULL,
                answer TEXT NOT NULL,
                INDEX idx_question (question(191))
            )
            """
        )
        self.connection.commit()
        self.logger.info("FAQ 表 jpkb 已就绪")

    def create_conversations_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(36) NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                INDEX idx_session_id (session_id)
            )
            """
        )
        self.connection.commit()
        self.logger.info("对话历史表 conversations 已就绪")

    def count_faqs(self):
        self.cursor.execute("SELECT COUNT(*) FROM jpkb")
        return int(self.cursor.fetchone()[0])

    @staticmethod
    def _read_faq(csv_path):
        data = pd.read_csv(csv_path).fillna("")
        columns = {"知识类别": "category", "问题": "question", "答案": "answer"}
        if set(columns).issubset(data.columns):
            return data.rename(columns=columns)
        raise ValueError("FAQ CSV 必须包含：知识类别、问题、答案")

    def insert_data(self, csv_path, replace=False):
        data = self._read_faq(csv_path)
        try:
            if replace:
                self.cursor.execute("DELETE FROM jpkb")
            rows = [
                (row.category.strip(), row.question.strip(), row.answer.strip())
                for row in data.itertuples(index=False)
                if row.question.strip() and row.answer.strip()
            ]
            self.cursor.executemany(
                "INSERT INTO jpkb (category, question, answer) VALUES (%s, %s, %s)",
                rows,
            )
            self.connection.commit()
            self.logger.info("已导入 %d 条 FAQ", len(rows))
            return len(rows)
        except Exception:
            self.connection.rollback()
            raise

    def fetch_questions(self):
        try:
            self.cursor.execute("SELECT question FROM jpkb ORDER BY id")
            return self.cursor.fetchall()
        except pymysql.MySQLError as exc:
            self.logger.error("查询 FAQ 问题失败: %s", exc)
            return []

    def fetch_answer(self, question):
        try:
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s LIMIT 1", (question,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except pymysql.MySQLError as exc:
            self.logger.error("查询 FAQ 答案失败: %s", exc)
            return None

    def close(self):
        self.cursor.close()
        self.connection.close()


if __name__ == "__main__":
    client = MySQLClient()
    client.create_table()
    faq_file = Path(__file__).resolve().parent.parent / "data" / "retail_faq.csv"
    client.insert_data(faq_file, replace=True)
    client.close()
