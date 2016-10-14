(ns ^{:doc    "Machine learning with Sparkling!"
      :author "Eric Weinstein <eric.q.weinstein@gmail.com>"}
euroclojure.sparkling
  (:require [sparkling.conf :as conf]
            [sparkling.core :as spark]
            [clojure.string :as s])
  (:import (org.apache.spark.mllib.tree DecisionTree)
           (org.apache.spark.ml.feature LabeledPoint)
           (org.apache.spark.mllib.linalg Vectors)))

;; Uncomment to test Sparkling code (otherwise, we get
;; `Only one SparkContext may be running in this JVM`)
(comment
  (defn make-spark-context
    "Creates the Apache Spark context using the Sparkling DSL."
    []
    (-> (conf/spark-conf)
        (conf/master "local")
        (conf/app-name "euroclojure")
        (spark/spark-context)))

  (def data
    (spark/map #(s/split % #" ") (spark/text-file (make-spark-context) "data/stop_data.txt")))

  (defn map-float
    "Converts a list of strings to a list of floats."
    [coll]
    (map #(Float/parseFloat %) coll))

  ;; TODO: Data cleanup/rename. (EQW 21 Oct 2016)
  (def dataset-lol
    (spark/map map-float data))

  (def dataset
    (spark/map
      (fn [[sex race stop-type post-stop-activity]]
        (let [pred (double-array [sex
                                  race
                                  stop-type])]
          ;; Spark requires that samples be packed into LabeledPoints.
          (LabeledPoint. post-stop-activity (Vectors/dense pred)))) dataset-lol))

  (comment
    ;; Side-effectful behavior (for performance)
    (spark/cache dataset)

    (def training
      (-> (spark/sample false 0.7 1234 dataset)
          (spark/cache)))

    (def testing
      (spark/subtract dataset training))

    (def categorical-features-info
      ;; Feature index 0: sex, two values (M/F)
      ;; Feature index 1: race, six values (Asian, Black, Hispanic, (American) Indian, White, Other)
      ;; Feature index 2: stop type, VEH (vehicle) or PED (pedestrian)
      (apply sorted-map (map int [0 2 1 6 2 2])))

    (def model
      ;; Max depth: 5
      ;; Max bins: 32
      (DecisionTree/trainClassifier training 2 categorical-features-info "gini" 5 32))

    (defn predict
      "Generates model predictions."
      [p]
      (let [prediction (.predict model (.features p))]
        [(.label p) prediction]))

    (def labels-and-preds
      (spark/map testing (fn [p] (predict p))))

    (def test-err
      (let [filtered (spark/filter labels-and-preds (fn [one two] (not (= one two))))]
        (/ (spark/count filtered) (spark/count testing)))))
)
