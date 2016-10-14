(defproject euroclojure "1.0.0-SNAPSHOT"
  :description "Code for EuroClojure 2016"
  :url "None"
  :license {:name "MIT" :url "TK"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [gorillalabs/sparkling "1.2.5"]
                 [yieldbot/flambo "0.7.2"]]
  :aot [#".*" sparkling.serialization sparkling.destructuring]
  :jvm-opts ["-Xmx1g"]
  :main euroclojure.core
  :profiles {:provided {:dependencies [[org.apache.spark/spark-core_2.10 "2.0.1"]
                                       [org.apache.spark/spark-mllib_2.10 "2.0.1"]
                                       [org.deeplearning4j/deeplearning4j-core "0.6.0"]
                                       [org.slf4j/slf4j-nop "1.7.12"]
                                       [org.nd4j/nd4j-native "0.6.0"]
                                       [com.fasterxml.jackson.core/jackson-annotations "2.6.0"]
                                       [com.fasterxml.jackson.core/jackson-core "2.6.0"]
                                       [com.fasterxml.jackson.core/jackson-databind "2.6.0"]
                                       ]}
             :dev {:plugins [[lein-dotenv "RELEASE"]]}})
