# CompressUtils 子包的对外入口。
#
# compress_function.py 放的是“无状态”的量化/反量化函数；
# compress_class.py 把这些函数包装成可保存状态的对象。模型的 past_key_value
# 会保存 CompressUnion 或 MixedPrecisionCompressUnion，而不是直接保存普通 tensor。

from .compress_class import CompressUnion, MixedPrecisionCompressUnion

__all__ = ["CompressUnion", "MixedPrecisionCompressUnion"]
