"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP

_TOOL_SUMMARIES = "\n".join(
    f"- {tool['name']}: {tool['description']}" for tool in TOOL_DEFINITIONS
)

SYSTEM_PROMPT = f"""Bạn là VinAssistant, trợ lý tư vấn của hệ sinh thái Vingroup.

## PERSONA
- Hỗ trợ khách hàng về sản phẩm VinFast, dịch vụ Vinpearl và yêu cầu hỗ trợ.
- Giao tiếp bằng tiếng Việt, thân thiện, chuyên nghiệp và súc tích.

## AVAILABLE TOOLS
{_TOOL_SUMMARIES}

## CORE RULES
- Chỉ khẳng định thông tin sản phẩm khi có dữ liệu từ tool hoặc dữ kiện đã nêu rõ.
- Không bịa giá, tình trạng hàng, chính sách, hoặc kết quả xử lý.
- Dùng công cụ phù hợp khi cần tra cứu sản phẩm hoặc tạo yêu cầu hỗ trợ.
- Chỉ báo đã tạo yêu cầu hỗ trợ khi tool trả về ticket thành công.
- Nếu thiếu thông tin cần thiết, hỏi khách hàng thay vì tự suy đoán.

## OPERATIONAL BOUNDARIES
- Chỉ tư vấn trong phạm vi sản phẩm, dịch vụ và hỗ trợ thuộc Vingroup.
- Lịch sự từ chối yêu cầu ngoài phạm vi này.

## OUTPUT CONTRACT
- Xử lý theo trình tự nội bộ: xác định ý định, gọi công cụ phù hợp, kiểm tra kết quả.
- Không hiển thị suy nghĩ nội bộ. Chỉ đưa ra câu trả lời cuối rõ ràng, dựa trên dữ liệu.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _extract_price(user_input: str):
        match = re.search(
            r"(?:\b(?:dưới|thấp hơn|tối đa|không quá|under|below|up to)\s*"
            r"|\bgiá\s*(?:(?:dưới|thấp hơn|tối đa|không quá)\s*)?)"
            r"(\d[\d.,]*)\s*(triệu|tr|million|m|tỷ|ty|billion|bn)?"
            r"|\b(\d[\d.,]*)\s*(triệu|tr|million|m|tỷ|ty|billion|bn)\b",
            user_input.lower(),
        )
        if not match:
            return None

        amount_text = match.group(1) or match.group(3)
        unit = match.group(2) or match.group(4)
        amount = int(amount_text.replace(".", "").replace(",", ""))
        multipliers = {
            "triệu": 1_000_000,
            "tr": 1_000_000,
            "million": 1_000_000,
            "m": 1_000_000,
            "tỷ": 1_000_000_000,
            "ty": 1_000_000_000,
            "billion": 1_000_000_000,
            "bn": 1_000_000_000,
        }
        return amount * multipliers.get(unit, 1)

    @staticmethod
    def _extract_customer_name(user_input: str):
        match = re.search(
            r"(?:tôi\s+tên(?:\s+là)?|tên\s+tôi\s+là|tên\s+là)\s+"
            r"([^,;.!?]+)",
            user_input,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_issue_description(user_input: str, customer_name: str):
        remaining = re.sub(
            r"^(?:.*?\b(?:tôi\s+tên(?:\s+là)?|tên\s+tôi\s+là|tên\s+là)\s+"
            + re.escape(customer_name)
            + r"\s*[,;:]?\s*)",
            "",
            user_input,
            count=1,
            flags=re.IGNORECASE,
        )
        if remaining == user_input:
            remaining = user_input

        remaining = re.sub(
            r"^(?:ghi nhận (?:phản hồi|yêu cầu)\s*:\s*)",
            "",
            remaining,
            flags=re.IGNORECASE,
        )
        remaining = re.sub(
            r"\s*(?:đây là )?(?:vấn đề )?(?:nghiêm trọng|trung bình|mức độ thấp|"
            r"cần xử lý gấp|khẩn cấp|mức độ trung bình|mức độ cao)[^.!?]*",
            "",
            remaining,
            flags=re.IGNORECASE,
        )
        remaining = remaining.split(".", maxsplit=1)[0].strip(" \t\r\n,;:-")
        remaining = re.sub(
            r"\b(phòng|xe)\s+tôi\b",
            r"\1",
            remaining,
            flags=re.IGNORECASE,
        )
        remaining = re.sub(r"\bcủa tôi\b", "", remaining, flags=re.IGNORECASE)
        remaining = re.sub(r"\s+", " ", remaining).strip()
        if remaining:
            return remaining[0].upper() + remaining[1:]
        return "Khách hàng cần được hỗ trợ."

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        normalized = user_input.lower()
        is_battery_warranty_faq = bool(
            re.search(r"bảo hành", normalized)
            and re.search(r"pin|ắc quy", normalized)
            and not re.search(
                r"bị lỗi|bị hỏng|hư hỏng|gặp vấn đề|sự cố|"
                r"ghi nhận|phản hồi|yêu cầu hỗ trợ|tạo ticket|cần hỗ trợ",
                normalized,
            )
        )
        category = None
        if re.search(r"resort|du lịch|vinpearl|khách sạn|đặt phòng", normalized):
            category = "du_lich"
        elif re.search(r"xe|vinfast|ô tô|ô-tô", normalized):
            category = "xe_dien"

        needs_catalog = not is_battery_warranty_faq and category is not None and bool(
            re.search(
                r"giá|bao nhiêu|dưới|thấp hơn|tối đa|không quá|xem|tìm|"
                r"sản phẩm|xe điện|resort|du lịch|vinpearl|khách sạn",
                normalized,
            )
        )
        customer_name = self._extract_customer_name(user_input)
        has_issue = bool(
            re.search(
                r"bị lỗi|bị hỏng|hư hỏng|ẩm mốc|gặp vấn đề|sự cố|"
                r"ghi nhận|phản hồi|yêu cầu hỗ trợ|tạo ticket|cần hỗ trợ",
                normalized,
            )
        )
        needs_ticket = customer_name is not None and has_issue

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "customer_name": customer_name,
            "category": category,
            "max_price": self._extract_price(user_input),
            "issue_description": (
                self._extract_issue_description(user_input, customer_name)
                if customer_name and needs_ticket
                else None
            ),
            "priority": (
                "high"
                if re.search(r"nghiêm trọng|xử lý gấp|khẩn cấp|ưu tiên cao", normalized)
                else "low"
                if re.search(r"mức độ thấp|ưu tiên thấp", normalized)
                else "medium"
            ),
            "is_vingroup_query": bool(
                re.search(r"vingroup|vinfast|vinpearl|xe điện|resort|du lịch", normalized)
            ),
            "is_battery_warranty_faq": is_battery_warranty_faq,
            "ticket_requested": bool(
                re.search(
                    r"ghi nhận|phản hồi|yêu cầu hỗ trợ|tạo ticket|cần hỗ trợ",
                    normalized,
                )
            ),
        }

    @staticmethod
    def _format_catalog_answer(products: List[Dict[str, Any]]) -> str:
        if not products:
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp với điều kiện bạn đưa ra."

        lines = ["Tôi tìm thấy các lựa chọn phù hợp:"]
        for product in products:
            price = f"{product['price_vnd']:,}".replace(",", ".")
            lines.append(f"- {product['name']}: {price} VNĐ.")
        return "\n".join(lines)

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intents(user_input)
        self.trace.append({"step": "init", "user_input": user_input})
        self.trace.append({"step": "intent_detection", "intents": intents})

        actions = []
        if intents["needs_catalog"]:
            catalog_args = {"category": intents["category"]}
            if intents["max_price"] is not None:
                catalog_args["max_price"] = intents["max_price"]
            actions.append(("search_product_catalog", catalog_args))
        if intents["needs_ticket"]:
            actions.append(
                (
                    "submit_support_ticket",
                    {
                        "customer_name": intents["customer_name"],
                        "issue_description": intents["issue_description"],
                        "priority": intents["priority"],
                    },
                )
            )

        observations = []
        iteration = 0
        for tool_name, arguments in actions:
            if iteration >= self.max_iterations:
                answer = "Không thể hoàn tất yêu cầu trong giới hạn số bước của agent."
                self.trace.append({"step": "limit_reached", "max_iterations": self.max_iterations})
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached",
                }

            iteration += 1
            self.trace.append(
                {
                    "step": "tool_call",
                    "iteration": iteration,
                    "tool": tool_name,
                    "arguments": arguments,
                }
            )
            observation = TOOL_MAP[tool_name](**arguments)
            self.trace.append(
                {
                    "step": "observation",
                    "iteration": iteration,
                    "tool": tool_name,
                    "result": observation,
                }
            )
            observations.append((tool_name, observation))

        answer_parts = []
        for tool_name, observation in observations:
            if tool_name == "search_product_catalog":
                answer_parts.append(self._format_catalog_answer(observation))
            else:
                answer_parts.append(
                    f"Yêu cầu hỗ trợ đã được tạo thành công. "
                    f"Mã ticket: {observation['ticket_id']} "
                    f"(khách hàng: {observation['customer_name']}, "
                    f"trạng thái: {observation['status']})."
                )

        if not answer_parts and intents["is_battery_warranty_faq"]:
            answer_parts.append(
                "Theo thông tin hiện có, VinFast VF 5 Plus được ghi nhận có "
                "bảo hành pin 10 năm. Thời hạn có thể khác theo mẫu xe; "
                "hãy kiểm tra chính sách chính thức cho đúng mẫu xe."
            )
        elif not answer_parts and intents["needs_ticket"] is False:
            if intents["ticket_requested"]:
                answer_parts.append(
                    "Vui lòng cung cấp tên và mô tả vấn đề cần hỗ trợ để tôi tạo "
                    "yêu cầu chính xác."
                )
            elif not intents["is_vingroup_query"]:
                answer_parts.append(
                    "Xin lỗi, tôi chỉ có thể hỗ trợ các sản phẩm, dịch vụ và yêu cầu "
                    "thuộc hệ sinh thái Vingroup."
                )
            else:
                answer_parts.append(
                    "Tôi chưa có đủ thông tin đã xác minh để trả lời chính xác câu hỏi này."
                )

        answer = "\n".join(answer_parts)
        self.trace.append({"step": "final_answer", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iteration or 1,
            "status": "completed",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
