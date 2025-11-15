# DB 기반 경로 캐싱 시스템

## 개요
기존의 `outputs/` 디렉토리에 이미지를 저장하던 방식에서, 데이터베이스 기반 캐싱 시스템으로 전환했습니다.
동일한 요청 파라미터로 경로 계산 요청이 들어오면 DB에서 캐시된 결과를 반환합니다.

## 주요 변경사항

### 1. DB 스키마 (`app/models.py`)
```python
class Route(Base):
    # 요청 파라미터 (캐시 키)
    t_start_idx, t_goal_idx
    lat_start, lon_start, lat_goal, lon_goal
    bcf, fuel_type
    w_fuel, w_bc, w_risk
    
    # 결과 데이터
    visualization_image: bytes  # 이미지를 DB에 직접 저장
    cost_summary_json: str      # JSON 형식으로 저장
    created_at: datetime
```

### 2. Repository (`app/repositoires.py`)
- `get_route()`: 11개 파라미터로 캐시 조회
- `add_route()`: 경로 결과를 이미지 바이트와 함께 저장

### 3. Service (`app/services.py`)
캐싱 로직:
1. DB에서 동일 파라미터 조회
2. **캐시 히트**: DB에서 이미지를 읽어 임시 파일로 저장 후 반환
3. **캐시 미스**: A* 경로 계산 → 시각화 생성 → DB 저장

## 사용 방법

### 1. DB 테이블 생성
```bash
python create_tables.py
```

### 2. 서버 실행
```bash
uvicorn app.main:app --reload
```

### 3. 캐싱 동작 테스트
```bash
python test_caching.py
```

## 캐싱 효과
- 동일한 요청 시 A* 경로 계산 스킵
- 응답 시간 대폭 단축
- DB 인덱스 추가 시 조회 성능 최적화 가능

## 향후 개선 사항
- `Route` 테이블에 복합 인덱스 추가 (캐시 키 필드들)
- 이미지를 S3 등 외부 스토리지로 분리 (DB 크기 최적화)
- TTL 기반 캐시 무효화
- 캐시 통계 모니터링
