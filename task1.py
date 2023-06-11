from pyspark import SparkContext
import sys, time
from itertools import combinations
from pyspark.sql import SparkSession
from graphframes import GraphFrame


# spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py 7 "/Users/leoli/Desktop/ub_sample_data.csv" "/Users/leoli/Desktop/task1_output.txt"
# spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 –-class task1 hw4.jar 7 "/Users/leoli/Desktop/ub_sample_data.csv" "/Users/leoli/Desktop/task1_output.txt"


if __name__ == "__main__":

    start_time = time.time()

    threshold = sys.argv[1]
    input_file_path = sys.argv[2]
    community_output_file_path = sys.argv[3]


    sc = SparkContext.getOrCreate()
    sparkSession = SparkSession(sc)
    sc.setLogLevel("ERROR")



    # print("----------------- read data ------------------")
    data_rdd = sc.textFile(input_file_path).filter(lambda x: "user_id" not in x)
    data_rdd = data_rdd.map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))

    user_group_businesses_rdd = data_rdd.groupByKey().mapValues(set)
    user_group_businesses_dict = user_group_businesses_rdd.collectAsMap()

    user_index_rdd = data_rdd.map(lambda x: x[0]).distinct().zipWithIndex()
    business_index_rdd = data_rdd.map(lambda x: x[1]).distinct().zipWithIndex()

    user_index_dict = user_index_rdd.collectAsMap()
    business_index_dict = business_index_rdd.collectAsMap()



    # print("----------------- generate pairs ------------------")
    pairs_rdd = sc.parallelize(list(combinations(user_index_dict.keys(), 2)))

    def find_edges(pair):
        user1_businesses = user_group_businesses_dict[pair[0]]
        user2_businesses = user_group_businesses_dict[pair[1]]
        common_businesses = user1_businesses.intersection(user2_businesses)
        return len(common_businesses) >= int(threshold)
    
    # edges_rdd = pairs_rdd.filter(find_edges).flatMap(lambda x: [x, (x[1], x[0])])
    edges_rdd = pairs_rdd.filter(find_edges)
    nodes_rdd = edges_rdd.flatMap(lambda x: x).distinct().map(lambda x: (x,))
    # print(nodes_rdd.take(10))

    edges_rdd = edges_rdd.flatMap(lambda x: [x, (x[1], x[0])])
    # print(edges_rdd.take(10))
 
    edges_df = edges_rdd.toDF(["src", "dst"])
    nodes_df = nodes_rdd.toDF(["id"])



    print("----------------- generate graph ------------------")
    graph = GraphFrame(nodes_df, edges_df)
    result = graph.labelPropagation(maxIter=5)
    output_rdd = result.rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).map(lambda x: sorted(list(x[1]))).sortBy(lambda x: (len(x), x[0]))
    output_list = output_rdd.collect()



    print("----------------- task 1 wrtie output file ------------------")
    with open(community_output_file_path, 'w', newline='') as output_file:
        for community in output_list:
            output_file.write(str(community)[1:-1] + "\n")
        

    print("time: {0:.5f}".format(time.time() - start_time))
