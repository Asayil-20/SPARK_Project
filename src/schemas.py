"""Schema definitions and alias dictionaries for SPARK."""

from __future__ import annotations

PRODUCT_ALIASES = {
    "product_id": ["product_id", "item_id", "sku", "productsku", "id"],
    "name": ["name", "product_name", "title", "product_title", "item_name"],
    "category": ["category", "main_category", "product_category"],
    "subcategory": ["subcategory", "sub_category", "subcat", "child_category"],
    "brand": ["brand", "manufacturer", "brand_name"],
    "price": ["price", "unit_price", "selling_price", "final_price", "list_price"],
    "description": ["description", "product_description", "details", "about"],
    "image_count": ["image_count", "images", "number_of_images", "image_cnt"],
    "image_url": ["image_url", "image", "main_image", "thumbnail", "thumbnail_url", "product_image"],
    "rating": ["rating", "avg_rating", "product_rating", "stars"],
    "review_count": ["review_count", "reviews_count", "num_reviews", "rating_count"],
    "created_at": ["created_at", "createdon", "created_date", "launch_date", "product_created_at"],
}

EVENT_ALIASES = {
    "event_id": ["event_id", "id", "interaction_id"],
    "user_id": ["user_id", "customer_id", "member_id", "visitor_id"],
    "session_id": ["session_id", "session", "visit_id", "browser_session_id"],
    "product_id": ["product_id", "item_id", "sku"],
    "event_type": ["event_type", "action", "interaction_type", "event_name"],
    "event_time": ["event_time", "timestamp", "event_timestamp", "event_date", "created_at"],
}

ORDER_ALIASES = {
    "order_id": ["order_id", "transaction_id", "purchase_id", "id"],
    "user_id": ["user_id", "customer_id", "buyer_id"],
    "order_date": ["order_date", "created_at", "purchase_date", "order_time", "date"],
    "total_amount": ["total_amount", "order_total", "grand_total", "amount", "revenue"],
    "status": ["status", "order_status", "state"],
}

ORDER_ITEMS_ALIASES = {
    "order_id": ["order_id", "transaction_id", "purchase_id"],
    "product_id": ["product_id", "item_id", "sku"],
    "quantity": ["quantity", "qty", "count"],
    "price": ["price", "unit_price", "selling_price", "amount", "item_price"],
    "unit_price": ["unit_price", "price", "selling_price", "list_price", "item_price"],
    "user_id": ["user_id", "customer_id", "buyer_id"],
}

REVIEW_ALIASES = {
    "product_id": ["product_id", "item_id", "sku"],
    "user_id": ["user_id", "customer_id", "author_id"],
    "rating": ["rating", "stars", "review_rating"],
    "review_text": ["review_text", "text", "comment", "review", "content"],
    "review_date": ["review_date", "created_at", "date"],
}

USERS_ALIASES = {
    "user_id": ["user_id", "customer_id", "member_id", "id"],
    "created_at": ["created_at", "signup_date", "registered_at"],
    "country": ["country", "country_name"],
    "city": ["city"],
    "gender": ["gender"],
    "age": ["age"],
}

PRODUCTS_REQUIRED = ["product_id"]
PRODUCTS_OPTIONAL_SAFE_DEFAULTS = {
    "name": "unknown_product",
    "category": "unknown_category",
    "subcategory": "unknown_subcategory",
    "brand": "unknown_brand",
    "price": 0.0,
    "description": "",
    "image_count": 0,
    "image_url": "",
    "rating": 0.0,
    "review_count": 0,
    "created_at": None,
}

EVENTS_REQUIRED = ["product_id", "event_type"]
EVENTS_OPTIONAL_SAFE_DEFAULTS = {
    "event_id": None,
    "user_id": None,
    "session_id": None,
    "event_time": None,
}

ORDERS_REQUIRED = ["order_id"]
ORDERS_OPTIONAL_SAFE_DEFAULTS = {
    "user_id": None,
    "order_date": None,
    "total_amount": 0.0,
    "status": "unknown",
}

ORDER_ITEMS_REQUIRED = ["order_id", "product_id"]
ORDER_ITEMS_OPTIONAL_SAFE_DEFAULTS = {
    "user_id": None,
    "quantity": 1,
    "price": 0.0,
    "unit_price": 0.0,
}

REVIEWS_REQUIRED = ["product_id"]
REVIEWS_OPTIONAL_SAFE_DEFAULTS = {
    "user_id": None,
    "rating": 0.0,
    "review_text": "",
    "review_date": None,
}

USERS_OPTIONAL_SAFE_DEFAULTS = {
    "user_id": None,
    "country": "",
    "city": "",
    "gender": "",
    "age": None,
}
