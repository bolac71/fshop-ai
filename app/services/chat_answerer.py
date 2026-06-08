import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.services.chat_planner import ChatPlan
from app.services.chat_tools import ToolResult


class ChatAnswerer:
    def __init__(self, llm_main=None):
        self.llm_main = llm_main
        self.prompt = ChatPromptTemplate.from_template(
            """
            Bạn là Finn, trợ lý mua sắm thời trang của FShop.

            Task: {task}
            Plan:
            {plan}

            Dữ liệu tool:
            {context}

            Câu hỏi khách: {question}

            Hướng dẫn:
            1. Trả lời tiếng Việt có dấu, tự nhiên, hữu ích.
            2. Nếu có sản phẩm trong dữ liệu tool, không nói rằng bạn không thể hiển thị sản phẩm.
            3. Với recommend_products, giới thiệu ngắn các sản phẩm phù hợp và nhắc rằng danh sách nằm bên dưới.
            4. Với product_advice, tư vấn dựa trên sản phẩm trong context, nói rõ nếu chỉ là gợi ý tham khảo.
            5. Với compare_products, so sánh theo nhu cầu của khách và chốt lựa chọn phù hợp nhất nếu đủ dữ liệu.
            6. Không bịa tồn kho, size, màu, giá hoặc chính sách không có trong context.
            """
        )

    def answer(self, question: str, plan: ChatPlan, tool_result: ToolResult) -> str:
        if tool_result.needs_clarification:
            return tool_result.clarification_question or "Bạn có thể nói rõ hơn nhu cầu của mình để Finn hỗ trợ chính xác hơn nhé."

        if not self.llm_main:
            return self.fallback(question, plan, tool_result)

        try:
            return (self.prompt | self.llm_main | StrOutputParser()).invoke(
                {
                    "task": plan.task,
                    "plan": json.dumps(plan.model_dump(), ensure_ascii=False),
                    "context": tool_result.context,
                    "question": question,
                }
            ).strip()
        except Exception as exc:
            print(f"CHAT ANSWERER | llm answer failed: {exc}")
            return self.fallback(question, plan, tool_result)

    def fallback(self, question: str, plan: ChatPlan, tool_result: ToolResult) -> str:
        if plan.task == "small_talk":
            return "Chào bạn, mình là Finn, trợ lý mua sắm của FShop. Bạn cần tìm sản phẩm, phối đồ, so sánh hay hỏi chính sách thì cứ nhắn mình nhé."

        if plan.task == "clarify":
            return plan.clarification_question or "Bạn muốn Finn tìm sản phẩm, tư vấn sản phẩm đang xem, so sánh hay hỏi chính sách/đơn hàng vậy?"

        names = [product.name for product in tool_result.products if product.name]
        if plan.task == "recommend_products":
            if names:
                return f"Mình tìm được vài lựa chọn phù hợp cho bạn: {', '.join(names[:3])}. Bạn xem các sản phẩm bên dưới nhé."
            return "Mình chưa tìm thấy sản phẩm thật sự phù hợp với tiêu chí này. Bạn có thể thử mở rộng màu, phong cách hoặc danh mục nhé."

        if plan.task == "product_advice":
            if names:
                return f"Với {names[0]}, mình có thể tư vấn ở mức tham khảo dựa trên thông tin sản phẩm. Nếu bạn cho mình biết thêm phong cách hoặc dịp mặc, mình sẽ gợi ý kỹ hơn."
            return "Mình chưa xác định được sản phẩm bạn đang hỏi. Bạn chọn lại sản phẩm hoặc gửi tên sản phẩm giúp Finn nhé."

        if plan.task == "compare_products":
            if len(names) >= 2:
                return f"Mình đang so sánh {', '.join(names[:3])}. Bạn có thể chọn theo nhu cầu chính như giá, phong cách, màu dễ phối hoặc size phù hợp."
            return "Mình cần ít nhất hai sản phẩm để so sánh. Bạn chọn thêm một sản phẩm nữa giúp Finn nhé."

        return "Mình đã nhận được yêu cầu của bạn, nhưng cần thêm một chút thông tin để hỗ trợ chính xác hơn nhé."
