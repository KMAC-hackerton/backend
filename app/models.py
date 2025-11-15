from database.common import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, Float, String, LargeBinary, Text
from sqlalchemy.dialects.mysql import MEDIUMBLOB
from datetime import datetime

class Route(Base):
    __tablename__ = "route"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # 요청 파라미터들 (캐시 키로 사용)
    t_start_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    t_goal_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    lat_start: Mapped[float] = mapped_column(Float, nullable=False)
    lon_start: Mapped[float] = mapped_column(Float, nullable=False)
    lat_goal: Mapped[float] = mapped_column(Float, nullable=False)
    lon_goal: Mapped[float] = mapped_column(Float, nullable=False)
    bcf: Mapped[float] = mapped_column(Float, nullable=False)
    fuel_type: Mapped[str] = mapped_column(String(50), nullable=False)
    w_fuel: Mapped[float] = mapped_column(Float, nullable=False)
    w_bc: Mapped[float] = mapped_column(Float, nullable=False)
    w_risk: Mapped[float] = mapped_column(Float, nullable=False)
    
    # 결과 데이터
    visualization_image: Mapped[bytes] = mapped_column(MEDIUMBLOB, nullable=False)
    cost_summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    
    # 메타데이터
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)