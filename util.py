{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "#util.py\n",
    "#!/usr/bin/env python\n",
    "# -*- coding: utf-8 -*-\n",
    "\"\"\"\n",
    "@Author: Yue Wang\n",
    "@Contact: yuewangx@mit.edu\n",
    "@File: util\n",
    "@Time: 4/5/19 3:47 PM\n",
    "\"\"\"\n",
    "\n",
    "\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn.functional as F\n",
    "\n",
    "\n",
    "def cal_loss(pred, gold, smoothing=True):\n",
    "    ''' Calculate cross entropy loss, apply label smoothing if needed. '''\n",
    "\n",
    "    gold = gold.contiguous().view(-1)\n",
    "\n",
    "    if smoothing:\n",
    "        eps = 0.2\n",
    "        n_class = pred.size(1)\n",
    "\n",
    "        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)\n",
    "        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)\n",
    "        log_prb = F.log_softmax(pred, dim=1)\n",
    "\n",
    "        loss = -(one_hot * log_prb).sum(dim=1).mean()\n",
    "    else:\n",
    "        loss = F.cross_entropy(pred, gold, reduction='mean')\n",
    "\n",
    "    return loss\n",
    "\n",
    "\n",
    "class IOStream():\n",
    "    def __init__(self, path):\n",
    "        self.f = open(path, 'a')\n",
    "\n",
    "    def cprint(self, text):\n",
    "        print(text)\n",
    "        self.f.write(text+'\\n')\n",
    "        self.f.flush()\n",
    "\n",
    "    def close(self):\n",
    "        self.f.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
