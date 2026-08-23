from backend.app.db.database import Base, engine
from backend.app.db import models


def main():
    Base.metadata.create_all(bind=engine)
    print("Database tables created")


if __name__ == "__main__":
    main()
