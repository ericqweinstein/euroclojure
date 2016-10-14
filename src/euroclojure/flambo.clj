(ns ^{:doc    "Machine learning with Flambo!"
      :author "Eric Weinstein <eric.q.weinstein@gmail.com>"}
  euroclojure.flambo
  (:require [flambo.conf :as conf]
            [flambo.api :as f]
            [clojure.string :as s])
  (:import (org.apache.spark.mllib.tree DecisionTree)
           (org.apache.spark.mllib.regression LabeledPoint)
           (org.apache.spark.mllib.linalg Vectors)))

(defn make-spark-context
  "Creates the Apache Spark context using the Flambo DSL."
  []
  (-> (conf/spark-conf)
      (conf/master "local")
      (conf/app-name "euroclojure")
      (f/spark-context)))

(def data
  (-> (f/text-file (make-spark-context) "data/stop_data.txt")
      (.zipWithIndex)
      (f/map f/untuple)
      (f/map (f/fn [[line _]]
                   (->> (s/split line #" ")
                        (map #(Float/parseFloat %)))))))

(def dataset
  (f/map data
    (f/fn [[sex race stop-type post-stop-activity]]
      (let [pred (double-array [sex
                                race
                                stop-type])]
           ;; Spark requires that samples be packed into LabeledPoints.
           (LabeledPoint. post-stop-activity (Vectors/dense pred))))))

;; Side-effectful behavior (for performance)
(f/cache dataset)

(def training
  (-> (f/sample dataset false 0.7 1234) ; Random seed
      (f/cache)))

(def testing
  (-> (.subtract dataset training)
      (f/cache)))

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
  (f/map testing (f/fn [p] (predict p))))

(def test-err
  (let [filtered (f/filter labels-and-preds (f/fn [[one two]] (not (= one two))))]
    (/ (f/count filtered) (f/count testing))))
