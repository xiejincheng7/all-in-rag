# 原示例代码保留在这里，便于和 partition_pdf 的写法对照。
# 原来的 partition 会根据文件类型自动选择对应的解析函数。
# from unstructured.partition.auto import partition
#
# # PDF文件路径
# pdf_path = "../../data/C2/pdf/rag.pdf"
#
# # 使用Unstructured加载并解析PDF文档
# elements = partition(
#     filename=pdf_path,
#     content_type="application/pdf"
# )
#
# # 打印解析结果
# print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")
#
# # 统计元素类型
# from collections import Counter
# types = Counter(e.category for e in elements)
# print(f"元素类型: {dict(types)}")
#
# # 显示所有元素
# print("\n所有元素:")
# for i, element in enumerate(elements, 1):
#     print(f"Element {i} ({element.category}):")
#     print(element)
#     print("=" * 60)


# 新示例：直接调用 PDF 解析函数，分别指定 hi_res 和 ocr_only 策略。
from collections import Counter

from unstructured.partition.pdf import partition_pdf


# PDF 文件路径（相对于 code/C2 作为工作目录）
pdf_path = "../../data/C2/pdf/rag.pdf"


def show_result(strategy: str) -> None:
    print(f"\n{'=' * 24} strategy={strategy} {'=' * 24}")

    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
        )
    except Exception as exc:
        # 某种策略所需依赖缺失时，打印原因并继续尝试另一种策略。
        print(f"解析失败: {type(exc).__name__}: {exc}")
        return

    total_chars = sum(len(str(element)) for element in elements)
    types = Counter(element.category for element in elements)
    print(f"解析完成: {len(elements)} 个元素, {total_chars} 字符")
    print(f"元素类型: {dict(types)}")

    # 展示各策略提取出的元素内容，便于比较结果。
    print("\n所有元素:")
    for i, element in enumerate(elements, 1):
        print(f"Element {i} ({element.category}):")
        print(element)
        print("=" * 60)


# 分别尝试高分辨率布局分析和纯 OCR 解析。
for parsing_strategy in ("hi_res", "ocr_only"):
    show_result(parsing_strategy)
