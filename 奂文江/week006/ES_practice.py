import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


es = Elasticsearch("http://localhost:9200")

# 检查连接
if es.ping():
    print("连接成功！")
else:
    print("连接失败！")

mapping = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "product_id":{
                "type": "integer"
            },
            "name": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "description": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "price": {
                "type": "float"
            },
            "category": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "brand":{
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "stock":{
                "type": "integer"
            },
            "tags":{
                "type": "keyword"
            },
            "rating":{
                "type": "float"
            },
            "created_at": {
                "type": "date"
            },
            "on_sale":{
                "type": "boolean"
            }
        }
    }
}

is_exists = es.indices.exists(index="products")

if is_exists:
    print('products已存在')
    es.indices.delete(index="products")
    print("删除produces索引")
    es.indices.refresh()
    #删除重建
    es.indices.create(index="products", body=mapping)
    print('products创建完成')
else:
    es.indices.create(index="products", body=mapping)
    print('products创建完成')

doc = {
        "product_id": 1001,
        "name": "Apple iPhone 14 Pro",
        "description": "6.1英寸智能手机，A16芯片，4800万像素主摄",
        "price": 7999.00,
        "category": "手机",
        "brand": "Apple",
        "stock": 150,
        "tags": ["智能手机", "苹果", "旗舰"],
        "rating": 4.8,
        "created_at": "2023-10-15T10:30:00",
        "on_sale": True
    }

#单个插入
res = es.index(index="products", document=doc)

print("单个插入",res)

es.indices.refresh(index="products")
result1 = es.search(index="products", body={"query": {"match_all": {}}})
print(result1['hits']['hits'])

print("========================================")
sample_products = [
    {
        "_index": "products",
        "product_id": 1002,
        "name": "华为 MateBook X Pro",
        "description": "13.9英寸笔记本电脑，3.1K触控全面屏，11代酷睿处理器",
        "price": 8999.00,
        "category": "电脑",
        "brand": "华为",
        "stock": 80,
        "tags": ["笔记本电脑", "轻薄本", "商务"],
        "rating": 4.7,
        "created_at": "2023-10-16T14:20:00",
        "on_sale": False
    },
    {
        "_index": "products",
        "product_id": 1003,
        "name": "索尼 WH-1000XM5 降噪耳机",
        "description": "无线蓝牙降噪耳机，30小时续航，AI降噪技术",
        "price": 2299.00,
        "category": "电子产品",
        "brand": "索尼",
        "stock": 200,
        "tags": ["耳机", "降噪", "无线"],
        "rating": 4.9,
        "created_at": "2023-10-17T09:15:00",
        "on_sale": True
    },
    {
        "_index": "products",
        "product_id": 1004,
        "name": "Apple iPhone 16 Pro",
        "description": "6.1英寸智能手机，A16芯片，4800万像素主摄",
        "price": 7999.00,
        "category": "手机",
        "brand": "Apple",
        "stock": 150,
        "tags": ["智能手机", "苹果", "旗舰"],
        "rating": 4.8,
        "created_at": "2023-10-15T10:30:00",
        "on_sale": True
    }
]
#批量插入数据
success_count, errors = bulk(es, sample_products, stats_only=False)

print("批量插入",success_count)
es.indices.refresh(index="products")
result2 = es.search(index="products", body={"query": {"match_all": {}}})
print(result2['hits']['hits'])
#条件过滤
result3 = es.search(index="products", body={"query": {"match": {"name": "索尼 WH-1000XM5 降噪耳机"}}})
print("条件过滤",result3['hits']['hits'])
