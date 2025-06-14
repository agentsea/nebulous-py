import time

from nebulous.data import Bucket

bucket = Bucket()
bucket.sync("./testdata", "s3://nebulous-rs/testdata/bucket-test")

time.sleep(2)

bucket.sync("s3://nebulous-rs/testdata/bucket-test", "./testdata-ret")
