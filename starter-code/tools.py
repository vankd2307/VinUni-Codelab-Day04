import json
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# TODO: Hoàn thiện hàm này — đọc file product_catalog.json, lọc theo category và max_price.
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    with open(catalog_file, "r", encoding="utf-8") as file:
        products = json.load(file)

    normalized_category = category.strip().lower()
    return [
        product
        for product in products
        if product["category"].lower() == normalized_category
        and product["price_vnd"] <= max_price
    ]


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# TODO: Hoàn thiện hàm này — tạo ticket mới và lưu vào support_tickets.json.
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    normalized_priority = priority.strip().lower()
    if normalized_priority not in {"low", "medium", "high"}:
        raise ValueError("priority must be 'low', 'medium', or 'high'")
    if not customer_name.strip():
        raise ValueError("customer_name must not be empty")
    if not issue_description.strip():
        raise ValueError("issue_description must not be empty")

    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as file:
            tickets = json.load(file)
        if not isinstance(tickets, list):
            raise ValueError("support_tickets.json must contain a JSON array")
    else:
        tickets = []

    now = datetime.now(timezone(timedelta(hours=7)))
    date_prefix = f"TK-{now.strftime('%Y%m%d')}-"
    sequence_numbers = [
        int(ticket["ticket_id"][len(date_prefix):])
        for ticket in tickets
        if isinstance(ticket, dict)
        and isinstance(ticket.get("ticket_id"), str)
        and ticket["ticket_id"].startswith(date_prefix)
        and ticket["ticket_id"][len(date_prefix):].isdigit()
    ]
    ticket_id = f"{date_prefix}{max(sequence_numbers, default=0) + 1:03d}"
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": normalized_priority,
        "status": "open",
        "created_at": now.isoformat(),
        "category": "general",
    }
    tickets.append(new_ticket)

    with open(tickets_file, "w", encoding="utf-8") as file:
        json.dump(tickets, file, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "priority": normalized_priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# TODO: Định nghĩa JSON Schema cho từng tool (name, description, parameters).
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' hoặc 'du_lich'.",
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ.",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": "Tạo yêu cầu hỗ trợ mới cho khách hàng Vingroup.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Tên khách hàng.",
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả vấn đề cần hỗ trợ.",
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên của yêu cầu.",
                    "enum": ["low", "medium", "high"],
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
