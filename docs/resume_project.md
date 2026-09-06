标题：基于miniSGLang的两级KV Cache压缩与前缀复用

项目简介：针对长上下文推理中活跃请求KV Cache与Prefix Cache竞争显存空间，导致PrefixCache频繁被淘汰的问题，基于mini-SGLang实现ZipCache两级KV Cache压缩，通过对冷Prefix Cache进行4bit/2bit压缩与缓存命中后的按需恢复，提高固定显存预算下的Prefix Cache有效容量和缓存命中率。

核心工作：
1. 两级缓存设计：设计FP/BF16 normal pool和低比特compressed pool，将请求结束后不再被引用的KV Cache压缩缓存至Compressed Pool；缓存命中后反量化至normal pool，保持原有page table语义和attention计算接口。
2. 量化与显存管理：基于K/V幅值评分，对单个请求中重要/非重要的token执行4bit/2bit混合量化，将量化值、参数、token位置打包存入Compressed Pool；实现Pool子空间分配、回收，管理压缩数据的存储与定位功能。
3. 推理链路接入：扩展radix tree节点信息以及Engine、CacheManager的调用链，实现对节点压缩状态的判断，实现请求结束后压缩、缓存命中反量化等功能。

项目成果：
