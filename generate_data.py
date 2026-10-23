import numpy as np
import pandas as pd

np.random.seed(42)

# 8 weeks total (56 days) to allow 4 weeks of baseline history
dates = pd.date_range(start="2026-08-01", periods=56, freq="D")
CAPACITY = 450

base_day_attendance = {
    0: 380,  # Monday
    1: 390,  # Tuesday
    2: 395,  # Wednesday
    3: 385,  # Thursday
    4: 310,  # Friday
    5: 230,  # Saturday
    6: 280,  # Sunday
}

menu_ratings = []
exam_deadlines = []
weekend_proximities = []
actual_attendances = []

for date in dates:
    day_of_week = date.weekday()

    is_weekend_prox = 1 if day_of_week in [4, 5] else 0
    weekend_proximities.append(is_weekend_prox)

    is_exam = 1 if 35 <= (date - dates[0]).days <= 40 else np.random.choice([0, 1], p=[0.90, 0.10])
    exam_deadlines.append(is_exam)

    if day_of_week in [2, 6]:
        menu_rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])
    else:
        menu_rating = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
    menu_ratings.append(menu_rating)

    base = base_day_attendance[day_of_week]
    menu_effect = (menu_rating - 3) * 12
    exam_effect = -35 if is_exam == 1 else 0
    weekend_effect = -45 if is_weekend_prox == 1 else 0
    noise = int(np.random.normal(0, 8))

    attendance = base + menu_effect + exam_effect + weekend_effect + noise
    attendance = min(CAPACITY, max(120, attendance))
    actual_attendances.append(attendance)

df = pd.DataFrame({
    "date": dates.strftime("%Y-%m-%d"),
    "day_of_week": dates.day_name(),
    "day_index": dates.weekday,
    "actual_attendance": actual_attendances,
    "menu_rating": menu_ratings,
    "exam_deadline": exam_deadlines,
    "weekend_proximity": weekend_proximities
})

df["baseline_4w_avg"] = np.nan

for idx, row in df.iterrows():
    target_date = dates[idx]
    past_4_dates = [target_date - pd.Timedelta(weeks=k) for k in range(1, 5)]
    past_entries = df[df["date"].isin([d.strftime("%Y-%m-%d") for d in past_4_dates])]

    if len(past_entries) == 4:
        df.at[idx, "baseline_4w_avg"] = round(past_entries["actual_attendance"].mean(), 2)

final_pilot_df = df.dropna(subset=["baseline_4w_avg"]).reset_index(drop=True)
cols = [
    "date", "day_of_week", "baseline_4w_avg",
    "menu_rating", "exam_deadline", "weekend_proximity", "actual_attendance"
]
final_pilot_df = final_pilot_df[cols]
final_pilot_df.to_csv("mess_attendance_pilot.csv", index=False)
print("SUCCESS: mess_attendance_pilot.csv generated successfully!")