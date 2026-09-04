from sqlalchemy.orm import Session
from . import models
from .executor import process_transaction


def next_batch_number(db: Session) -> int:
    meta = db.query(models.Meta).filter(models.Meta.id == 1).first()
    if not meta:
        meta = models.Meta(id=1, batch_counter=0)
        db.add(meta)
    meta.batch_counter += 1
    db.commit()
    db.refresh(meta)
    return meta.batch_counter


def count_open(db: Session) -> int:
    return (
        db.query(models.Transaction)
        .filter(models.Transaction.status.in_(["new", "in_progress"]))
        .count()
    )


def run_batch(db: Session, limit: int = 1000) -> dict:
    current_batch = next_batch_number(db)
    txns = (
        db.query(models.Transaction)
        .filter(models.Transaction.status.in_(["new", "in_progress"]))
        .limit(limit)
        .all()
    )
    processed = 0
    for txn in txns:
        process_transaction(db, txn, current_batch)
        processed += 1
    return {"processed": processed, "batch_number": current_batch, "remaining_open": count_open(db)}