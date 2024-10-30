from prefect import flow, task
import dummy

flow(log_prints=True)(dummy.dummy)