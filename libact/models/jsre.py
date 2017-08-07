"""JSRE

An interface for jSRE Classifier model.
"""
import sys
import os
import subprocess
import re
import numpy as np
import logging
import datetime
import time
import random
LOGGER = logging.getLogger(__name__)

from libact.base.interfaces import ProbabilisticModel


class JSRE(ProbabilisticModel):

    """Wrapper for jSRE classifier utilizing a SVM with a shallow linguistics kernel

    References
    ----------
    https://hlt-nlp.fbk.eu/technologies/jsre

    """

    def __init__(self, jsre_path, tmp_dir='/tmp', cache_size=1024, n_gram=3, window_size=2, C=None, max_memory=2048, kernel='SL'):
        """model_path => where to save the model to"""
        self.tmp_dir = tmp_dir
        self.model = self.__get_tmp_filename('model')
        self.tmp_training_file = self.__get_tmp_filename('train')
        self.tmp_test_file = self.__get_tmp_filename('test')
        self.tmp_output_file = self.__get_tmp_filename('output')
        self.classpath = './bin:./lib/*'
        self.jsre_path = jsre_path
        self.max_memory = '-mx{}M'.format(max_memory)
        self.jsre_paras = '-m {} -k {} -n {} -w {}'.format(cache_size, kernel, n_gram, window_size) + \
                          (' -c {}'.format(C) if C is not None else '')
        self.predict_template = 'java -cp {cp} {memory} org.itc.irst.tcc.sre.Predict {to_predict} {model} {output}'
        self.train_template = 'java -cp {cp} {memory} org.itc.irst.tcc.sre.Train {jsre_paras} {to_train} {model_output}'

    def __generate_timestamp(self):
        return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    def __get_tmp_filename(self, string):
        ts = self.__generate_timestamp()
        rnd = ''.join(random.choices('abcdfeghijklmnopqrstuvwxyz0123456789', k=5))
        return os.path.join(self.tmp_dir, '{}-{}-{}.jsre'.format(string, ts, rnd))

    def __run_command(self, cmd, output_file=None):
        # print('Running "{}"'.format(cmd))
        try:
            pc = subprocess.run(cmd.split(' '), check=True, stdout=subprocess.PIPE,
                                cwd=self.jsre_path)
            if output_file is None:
                return None, pc.stdout.decode('utf-8')
            with open(output_file, 'r') as outputf:
                predictions = outputf.read().strip().split('\n')
            return predictions, pc.stdout
        except subprocess.CalledProcessError as e:
            LOGGER.error('Could not run "{}": {}'.format(cmd, e))
            sys.exit(-1)

    def __raw_predict(self, instances):
        with open(self.tmp_test_file, 'w') as testf:
            for instance in instances:
                # use 1 as default label b/c -1 will crash jSRE even tough thats the proposed label for unknown
                testf.write('1\t{}\n'.format(instance[0]))  # all features are encoded as one for jSRE

        cmd = self.predict_template.format(
            cp=self.classpath, memory=self.max_memory, model=self.model,
            to_predict=self.tmp_test_file, output=self.tmp_output_file)

        raw_prediction, _ = self.__run_command(cmd, self.tmp_output_file)
        splitted_lines = (line.split('\t') for line in raw_prediction[1:])
        predictions, probas = zip(*[(int(float(row[0])), [float(n) for n in row[2:]]) for row in splitted_lines])
        try:
            os.remove(self.tmp_test_file)
            os.remove(self.tmp_output_file)
        except OSError:
            pass
        return predictions, probas

    def train(self, dataset, *args, **kwargs):
        lines = ['{}\t{}'.format(lbl, feat) for feat, lbl in zip(*dataset.format_jsre())]
        with open(self.tmp_training_file, 'w') as trainingf:
            trainingf.write('\n'.join(lines))

        cmd = self.train_template.format(
            cp=self.classpath, memory=self.max_memory, model_output=self.model,
            to_train=self.tmp_training_file, jsre_paras=self.jsre_paras)

        self.__run_command(cmd)
        try:
            os.remove(self.tmp_training_file)
        except OSError:
            pass

    def predict(self, instances, *args, **kwargs):
        predictions, _ = self.__raw_predict(instances)
        return np.array(predictions)

    def predict_proba(self, instances, *args, **kwargs):
        _, probas = self.__raw_predict(instances)
        # this is dependent on the data and can be the other way round as well!
        return np.array(probas)

    def score(self, dataset, *args, **kwargs):
        lines = ['{}\t{}'.format(lbl, feat) for feat, lbl in zip(*dataset.format_jsre())]
        with open(self.tmp_test_file, 'w') as trainingf:
            trainingf.write('\n'.join(lines))

        cmd = self.predict_template.format(
            cp=self.classpath, memory=self.max_memory, model=self.model,
            to_predict=self.tmp_test_file, output=self.tmp_output_file)
        _, stdout = self.__run_command(cmd)

        match = re.search(r'Accuracy = (\d+\.\d+)%', stdout)
        if match is None or len(match.groups()) < 1:
            LOGGER.error('Could not extract accuracy from output of "{}"'.format(cmd))
            sys.exit(-1)

        accuracy = float(match.groups()[0])/100
        try:
            os.remove(self.tmp_test_file)
            os.remove(self.tmp_output_file)
        except OSError:
            pass

        return accuracy