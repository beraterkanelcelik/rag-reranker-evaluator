import asyncio

from backend.core.database import SessionLocal
from backend.services.dataset_ingestion import DatasetIngestionService


async def main() -> None:
    async with SessionLocal() as session:
        service = DatasetIngestionService(session)
        result = await service.ingest("official/pdf/arxiv")
        counts = await service.get_counts()
        print(result)
        print(counts)


if __name__ == "__main__":
    asyncio.run(main())
