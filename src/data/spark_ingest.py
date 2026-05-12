from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp
import argparse

def process_domain(spark: SparkSession, domain_path: str, output_path: str) -> None:
    df = spark.read.json(domain_path)
    
    clean_df = df.filter(col("item_id").isNotNull()) \
                 .withColumn("ingested_at", current_timestamp()) \
                 .dropDuplicates(["item_id"])
                 
    clean_df.write.mode("overwrite").parquet(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    
    spark = SparkSession.builder \
        .appName("HermesDataIngestion") \
        .config("spark.executor.memory", "16g") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
        
    process_domain(spark, args.domain_path, args.output_path)
    spark.stop()
