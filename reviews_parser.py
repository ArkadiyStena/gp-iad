import re
import time

import numpy as np
import pandas as pd


nil_values = {"", "nil", "no", "none", "nan", "na", "n/a", "нет", "нет отзыва", "без отзыва", "ничего"}

delivery_words = [
    r"достав", r"курьер", r"прив[её]з", r"прин[её]с", r"заказ.*ехал", r"еда.*ехала",
    r"ждал", r"ожидан", r"опозда", r"задерж", r"вовремя", r"быстр", r"долг[оаиую]",
    r"deliver", r"courier", r"rider", r"wait", r"late", r"delay", r"fast", r"quick", r"slow", r"on time",
]

positive_words = [
    r"быстр", r"оператив", r"вовремя", r"мгнов", r"не пришлось ждать", r"без задерж", r"не задерж",
    r"fast", r"quick", r"on time", r"no delay", r"without delay", r"didn.t wait",
]

negative_words = [
    r"долг[оаиую]", r"опозда", r"задерж", r"ждал", r"не дожд", r"холодн", r"остыл", r"кошмар", r"ужас",
    r"late", r"delay", r"slow", r"wait", r"cold", r"too long", r"not on time", r"never arrived",
]

# время числом: "25 мин", "1.5 часа"
minute_re = re.compile(r"(?<!\d)(\d{1,3})(?:[\.,](\d))?\s*(?:минут|мин\.?|минута|минуты|minutes?|mins?|min\.?)(?!\w)")
hour_re = re.compile(r"(?<!\d)(\d{1,2})(?:[\.,](\d))?\s*(?:час|часа|часов|hours?|hrs?|h)(?!\w)")

# время словами — переводим в минуты
word_time = [
    (r"полтора часа", 90),
    (r"полторы часа", 90),
    (r"полчаса", 30),
    (r"пол часа", 30),
    (r"\bчас\b", 60),
    (r"one hour", 60),
    (r"half an hour", 30),
    (r"an hour", 60),
]


def count_matches(patterns, text):
    """Сколько шаблонов из списка встретилось в тексте хотя бы раз"""
    return sum(bool(re.search(pattern, text)) for pattern in patterns)


def find_minutes(text):
    """Самое большое время доставки, упомянутое в тексте, в минутах (или NaN)"""
    times = []
    for match in minute_re.finditer(text):
        times.append(float(match.group(1)))
    for match in hour_re.finditer(text):
        hours = float(match.group(1) + ("." + match.group(2) if match.group(2) else ""))  # 
        times.append(hours * 60)
    for pattern, minutes in word_time:
        if re.search(pattern, text):
            times.append(minutes)
    # берём максимум, т.к. долгое ожидание важнее короткого
    return max(times) if times else np.nan


def sentiment_by_delivery(has_time, minutes, positive, negative):
    """Сантимент по доставке. Названное время перебивает словарь."""
    if negative > positive or (has_time and minutes >= 60):
        return "negative", -1
    if positive > negative or (has_time and minutes <= 30):
        return "positive", 1
    return "neutral", 0


def delivery_review_parser(text):
    clean = "" if pd.isna(text) else str(text).lower().replace("ё", "е")
    clean = re.sub(r"\s+", " ", clean).strip()  # замена любой последовательности пробельных символов на один пробел

    result = {
        "review_has_text": int(clean not in nil_values),
        "mentions_delivery_time": 0,
        "delivery_time_minutes": np.nan,
        "delivery_time_sentiment": "no_mention" if clean not in nil_values else "no_review",
        "delivery_time_score": 0,
        "delivery_time_complaint": 0,
        "delivery_time_praise": 0,
    }
    if clean in nil_values:
        return pd.Series(result)

    minutes = find_minutes(clean)
    has_time = pd.notna(minutes)
    if not (count_matches(delivery_words, clean) or has_time):
        return pd.Series(result)

    sentiment, score = sentiment_by_delivery(
        has_time, minutes, count_matches(positive_words, clean), count_matches(negative_words, clean)
    )
    result.update({
        "mentions_delivery_time": 1,
        "delivery_time_minutes": minutes,
        "delivery_time_sentiment": sentiment,
        "delivery_time_score": score,
        "delivery_time_complaint": int(score == -1),
        "delivery_time_praise": int(score == 1),
    })
    return pd.Series(result)


apps = {
    "ru.foodfox.client": ("Яндекс Еда", "yandex_food_google_play_reviews_parsed.csv"),
    "com.deliveryclub": ("Delivery Club", "delivery_club_google_play_reviews_parsed.csv"),
}

target_count = 3000
max_raw_reviews = 15000


def collect_reviews(app_id, output_path):
    from google_play_scraper import Sort, reviews 

    kept = []
    seen_ids = set()
    token = None
    raw = 0

    while sum(len(part) for part in kept) < target_count and raw < max_raw_reviews:
        batch, token = reviews(app_id, lang="ru", country="ru", sort=Sort.NEWEST, count=200, continuation_token=token)
        if not batch:
            break
        raw += len(batch)

        df = pd.DataFrame(batch).rename(columns={"content": "review", "at": "review_date", "score": "rating"})
        df = df[["reviewId", "review_date", "rating", "thumbsUpCount", "review"]]
        df = df[~df["reviewId"].isin(seen_ids)]  # строки у которых reviewId не находится в seen
        seen_ids.update(df["reviewId"])

        features = df["review"].apply(delivery_review_parser)  # применяем парсер ко всем отзывам в батче
        df = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)  # добавляем синтетические признаки в датасет (axis=1 == склеиваем по горизонтали)
        about_delivery = df[df["mentions_delivery_time"] == 1]  # оставляем только отзывы связанные с доставкой
        if len(about_delivery):
            kept.append(about_delivery)

        print(f"Всего отзывов: {raw}; оставили: {sum(len(part) for part in kept)}")
        if token is None:
            break
        time.sleep(0.35)

    result = pd.concat(kept, ignore_index=True).head(target_count)  # соединяем все батчи в один датасет
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"{output_path}: сохранено {len(result)} отзывов про доставку\n")
    return result


if __name__ == "__main__":
    for app_id, (name, path) in apps.items():
        print(f"Сбор отзывов для: {name}")
        collect_reviews(app_id, path)
