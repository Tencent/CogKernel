from sqlalchemy import Column, Integer, String, Boolean, Index, create_engine, desc, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dateutil.relativedelta import relativedelta, MO
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

# Database connection with connection pooling
def get_db_connector(db_name):
    POSTGRES_HOST = "postgres"
    POSTGRES_PORT = "5432"
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    return create_engine(
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{db_name}",
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=3600,
        echo=False,  # Set to True for debugging
    )

POSTGRES_DB_USER = "cognitive_kernel_user"
POSTGRES_DB_HISTORY = "cognitive_kernel_history"
POSTGRES_DB_RAW_DATA = "cognitive_kernel_raw_data"
POSTGRES_DB_ANNOTATION = "cognitive_kernel_annotation"

user_db_connector = get_db_connector(POSTGRES_DB_USER)
history_db_connector = get_db_connector(POSTGRES_DB_HISTORY)
raw_data_db_connector = get_db_connector(POSTGRES_DB_RAW_DATA)
annotation_db_connector = get_db_connector(POSTGRES_DB_ANNOTATION)

# Models with indexes and JSONB
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, index=True)
    password = Column(String)

    def __repr__(self):
        return f"<User(name='{self.username}')>"

class MessageHistory(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        Index('idx_session_username', 'session_id', 'username'),
        Index('idx_updated_time', 'updated_time'),
    )
    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    model_name = Column(String)
    username = Column(String, index=True)
    initial_message = Column(String)
    messages = Column(JSONB)
    created_time = Column(String)
    updated_time = Column(String)
    archived = Column(Boolean, default=False)

    def __repr__(self):
        return f"<MessageHistory(session_id='{self.session_id}', username='{self.username}')>"

class RawData(Base):
    __tablename__ = "annotations_by_session"
    __table_args__ = (
        Index('idx_rawdata_session_message', 'session_id', 'message_id'),
    )
    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    message_id = Column(String, index=True)
    username = Column(String, index=True)
    raw_data = Column(JSONB)
    created_time = Column(String)
    updated_time = Column(String)

    def __repr__(self):
        return f"<RawData(session_id='{self.session_id}', username='{self.username}')>"

class Annotation(Base):
    __tablename__ = "annotations_by_message"
    __table_args__ = (
        Index('idx_annotation_session_message', 'session_id', 'message_id'),
        Index('idx_annotation_username', 'username'),
        Index('idx_annotation_updated_time', 'updated_time'),
    )
    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    message_id = Column(String, index=True)
    username = Column(String, index=True)
    tag = Column(String)
    for_evaluation = Column(Boolean)
    old_message = Column(String)
    suggestion = Column(String)
    annotations = Column(JSONB)
    created_time = Column(String)
    updated_time = Column(String)

    def __repr__(self):
        return f"<Annotation(session_id='{self.session_id}', username='{self.username}')>"

# Create tables
User.metadata.create_all(user_db_connector)
MessageHistory.metadata.create_all(history_db_connector)
RawData.metadata.create_all(raw_data_db_connector)
Annotation.metadata.create_all(annotation_db_connector)

# Session makers
UserSessionLocal = sessionmaker(bind=user_db_connector)
MessageHistorySessionLocal = sessionmaker(bind=history_db_connector)
RawDataSessionLocal = sessionmaker(bind=raw_data_db_connector)
AnnotationSessionLocal = sessionmaker(bind=annotation_db_connector)

# --- CRUD Functions with Error Handling and Logging ---

def update_or_create_session(session_id, username, model_name, messages, updated_time):
    session = MessageHistorySessionLocal()
    try:
        logger.info(f"Updating/creating session {session_id} for user {username}.")
        existing_session = (
            session.query(MessageHistory)
            .filter_by(session_id=session_id, username=username)
            .first()
        )
        if existing_session:
            existing_session.messages = json.dumps(messages)
            existing_session.updated_time = updated_time
        else:
            new_session = MessageHistory(
                session_id=session_id,
                model_name=model_name,
                username=username,
                initial_message=messages[0]["message"][0]["content"],
                messages=json.dumps(messages),
                created_time=updated_time,
                updated_time=updated_time,
                archived=False,
            )
            session.add(new_session)
        session.commit()
        logger.info("Session updated or created successfully.")
    except Exception as e:
        logger.error(f"Error updating/creating session: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def get_sessions_by_username(model_name: str, username: str):
    session = MessageHistorySessionLocal()
    try:
        sessions = (
            session.query(MessageHistory)
            .filter_by(username=username, model_name=model_name, archived=False)
            .order_by(desc(MessageHistory.updated_time))
            .limit(50)
            .with_entities(
                MessageHistory.id,
                MessageHistory.session_id,
                MessageHistory.initial_message,
                MessageHistory.updated_time,
            )
            .all()
        )
        sessions_info = [
            {
                "id": s.id,
                "session_id": s.session_id,
                "initial_message": s.initial_message,
                "updated_time": s.updated_time,
            }
            for s in sessions
        ]
        return sessions_info
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        raise
    finally:
        session.close()

def get_session_by_id(session_id: int):
    session = MessageHistorySessionLocal()
    try:
        session_data = session.query(MessageHistory).filter_by(id=session_id).first()
        if session_data:
            session_info = {
                "id": session_data.id,
                "session_id": session_data.session_id,
                "username": session_data.username,
                "model_name": session_data.model_name,
                "initial_message": session_data.initial_message,
                "updated_time": session_data.updated_time,
                "messages": session_data.messages,
            }
        else:
            session_info = None
        return session_info
    except Exception as e:
        logger.error(f"Error fetching session by id: {e}")
        raise
    finally:
        session.close()

def archive_session_by_id(session_id: int):
    session = MessageHistorySessionLocal()
    try:
        session_data = session.query(MessageHistory).filter_by(id=session_id).first()
        if session_data:
            session_data.archived = True
            session.commit()
            logger.info(f"Session {session_id} archived.")
    except Exception as e:
        logger.error(f"Error archiving session: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def update_or_create_rawdata(session_id, message_id, username, messages_in_train_format, updated_time):
    session = RawDataSessionLocal()
    try:
        existing_message_raw_data = (
            session.query(RawData)
            .filter_by(session_id=session_id, message_id=message_id, username=username)
            .first()
        )
        if existing_message_raw_data:
            existing_message_raw_data.raw_data = json.dumps(messages_in_train_format)
            existing_message_raw_data.updated_time = updated_time
        else:
            new_message_raw_data = RawData(
                session_id=session_id,
                message_id=message_id,
                username=username,
                raw_data=json.dumps(messages_in_train_format),
                created_time=updated_time,
                updated_time=updated_time,
            )
            session.add(new_message_raw_data)
        session.commit()
        logger.info("Raw data updated or created successfully.")
    except Exception as e:
        logger.error(f"Error updating/creating raw data: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def get_rawdata_by_message_id(message_id: str):
    logger.info(f"Fetching raw data for message_id: {message_id}")
    session = RawDataSessionLocal()
    try:
        raw_datas = (
            session.query(RawData)
            .filter_by(message_id=message_id)
            .with_entities(RawData.session_id, RawData.message_id, RawData.raw_data)
            .all()
        )
        raw_data_info = [
            {
                "session_id": raw_data.session_id,
                "message_id": raw_data.message_id,
                "raw_data": raw_data.raw_data,
            }
            for raw_data in raw_datas
        ]
        return raw_data_info
    except Exception as e:
        logger.error(f"Error fetching raw data by message_id: {e}")
        raise
    finally:
        session.close()

def get_rawdata_by_session_id(session_id: str):
    session = RawDataSessionLocal()
    try:
        raw_datas = (
            session.query(RawData)
            .filter_by(session_id=session_id)
            .with_entities(RawData.session_id, RawData.message_id, RawData.raw_data)
            .all()
        )
        raw_data_info = [
            {
                "session_id": raw_data.session_id,
                "message_id": raw_data.message_id,
                "raw_data": raw_data.raw_data,
            }
            for raw_data in raw_datas
        ]
        return raw_data_info
    except Exception as e:
        logger.error(f"Error fetching raw data by session_id: {e}")
        raise
    finally:
        session.close()

def update_or_create_annotation(
    session_id,
    message_id,
    username,
    tag,
    for_evaluation,
    old_message,
    suggestion,
    messages_in_train_format,
    updated_time,
):
    session = AnnotationSessionLocal()
    try:
        existing_message_annotation = (
            session.query(Annotation)
            .filter_by(session_id=session_id, message_id=message_id, username=username)
            .first()
        )
        if existing_message_annotation:
            existing_message_annotation.tag = tag
            existing_message_annotation.for_evaluation = for_evaluation
            existing_message_annotation.old_message = old_message
            existing_message_annotation.suggestion = suggestion
            existing_message_annotation.annotations = json.dumps(messages_in_train_format)
            existing_message_annotation.updated_time = updated_time
        else:
            new_message_annotation = Annotation(
                session_id=session_id,
                message_id=message_id,
                username=username,
                tag=tag,
                for_evaluation=for_evaluation,
                old_message=old_message,
                suggestion=suggestion,
                annotations=json.dumps(messages_in_train_format),
                created_time=updated_time,
                updated_time=updated_time,
            )
            session.add(new_message_annotation)
        session.commit()
        user_record_count = session.query(Annotation).filter_by(username=username).count()
        logger.info(f"Annotation updated or created for user {username}. Total records: {user_record_count}.")
    except Exception as e:
        logger.error(f"Error updating/creating annotation: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def extract_annotation_info(annotations):
    return [
        {
            "session_id": annotation.session_id,
            "message_id": annotation.message_id,
            "username": annotation.username,
            "tag": annotation.tag,
            "for_evaluation": annotation.for_evaluation,
            "old_message": annotation.old_message,
            "suggestion": annotation.suggestion,
            "annotations": annotation.annotations,
            "created_time": annotation.created_time,
            "updated_time": annotation.updated_time,
        }
        for annotation in annotations
    ]

def get_all_annotations():
    session = AnnotationSessionLocal()
    try:
        annotations = (
            session.query(Annotation)
            .order_by(Annotation.updated_time.desc())
            .all()
        )
        annotations_info = extract_annotation_info(annotations)
        logger.info(f"Fetched {len(annotations_info)} annotations.")
        return annotations_info
    except Exception as e:
        logger.error(f"Error fetching all annotations: {e}")
        raise
    finally:
        session.close()

def get_anno_by_username_and_date_range(username: str, start_date: str, end_date: str):
    session = AnnotationSessionLocal()
    try:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        annotations = (
            session.query(Annotation)
            .filter(
                Annotation.username == username,
                func.date(Annotation.updated_time) >= start_date_obj,
                func.date(Annotation.updated_time) <= end_date_obj,
            )
            .order_by(Annotation.updated_time.desc())
            .all()
        )
        annotations_info = extract_annotation_info(annotations)
        logger.info(f"Fetched {len(annotations_info)} annotations for user {username}.")
        return annotations_info
    except Exception as e:
        logger.error(f"Error fetching annotations by date range: {e}")
        raise
    finally:
        session.close()

def get_annotations_for_evaluation():
    session = AnnotationSessionLocal()
    try:
        annotations = (
            session.query(Annotation)
            .filter(Annotation.for_evaluation == True)
            .order_by(Annotation.updated_time.desc())
            .all()
        )
        annotations_info = extract_annotation_info(annotations)
        logger.info(f"Fetched {len(annotations_info)} annotations for evaluation.")
        return annotations_info
    except Exception as e:
        logger.error(f"Error fetching annotations for evaluation: {e}")
        raise
    finally:
        session.close()

def get_annotation_counts_by_username(username: str, current_time: str):
    session = AnnotationSessionLocal()
    try:
        today = datetime.strptime(current_time, "%Y-%m-%dT%H:%M:%S.%fZ").date()
        this_week_start = today - relativedelta(weekday=MO(-1))
        count_today = (
            session.query(Annotation)
            .filter(
                Annotation.username == username,
                func.date(Annotation.updated_time) >= today,
            )
            .count()
        )
        count_this_week = (
            session.query(Annotation)
            .filter(
                Annotation.username == username,
                func.date(Annotation.updated_time) >= this_week_start,
            )
            .count()
        )
        total_count = (
            session.query(Annotation)
            .filter(Annotation.username == username)
            .count()
        )
        return {
            "today": count_today,
            "this_week": count_this_week,
            "total": total_count,
        }
    except Exception as e:
        logger.error(f"Error fetching annotation counts: {e}")
        raise
    finally:
        session.close()
