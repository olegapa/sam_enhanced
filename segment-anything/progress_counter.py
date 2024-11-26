from container_status import ContainerStatus
import logging

logging.basicConfig(level=logging.INFO, filename='/output/container_status.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProgressCounter:
    def __init__(self, total, processed, cs: ContainerStatus):
        self.total = total
        self.cs = cs
        self.processed = processed
        self.logger = logger

    def report_status(self, report_amount, stage, out_file=None, test_error=None, train_error=None):
        self.processed += report_amount
        data = {"stage": f"{stage} из 3", "progress": round(100*(self.processed / self.total), 2)}
        if out_file or test_error or train_error:
            data['statistics'] = dict()
            if out_file:
                data['statistics']['out_file'] = out_file
            if train_error:
                data['statistics']['train_error'] = test_error
            if test_error:
                data['statistics']['test_error'] = test_error
        self.logger.info(f'Reporting data: {data}')
        self.cs.post_progress(data)
