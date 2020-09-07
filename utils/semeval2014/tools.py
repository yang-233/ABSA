from typing import *

import utils.semeval2014.base as sb
import xml.etree.ElementTree as ET
import copy
from sklearn import metrics

#尽量保持原代码不变

polarity2int = {"neutral" : 0, "positive" : 1, "negative" : 2}
int2polarity = ("neutral", "positive", "negative")

class Instancev2(sb.Instance):
    def __init__(self, id:str, text:str): #　新增一个构造函数
        self.text = text
        self.id = id
        self.aspect_terms = []
        self.aspect_categories = []

class Corpusv2(sb.Corpus): # 仅仅用于保存xml文件
    def __init__(self):
        pass
    
class ATSAItems(object): # 用于ATSA任务的项
    def __init__(self, data:List[tuple]):
        self.id = []
        self.text = []
        self.term = []
        self.polarity = []
        self.size = len(data)
        for i in data:
            self.id.append(i[0])
            self.text.append(i[1])
            self.term.append(i[2])
            self.polarity.append(i[3])

    def to_instances(self):
        instances = []
        for i in range(self.size):
            if len(instances) == 0 or self.id[i] != instances[-1].id: # 需要添加新的instance
                instances.append(Instancev2(self.id[i], self.text[i]))
            instances[-1].add_aspect_term(self.term[i], self.polarity[i])
        return instances
    
    def __getitem__(self, idx):
        return self.text[idx], self.term[idx], self.polarity[idx]

    def get_y(self):
        return [polarity2int[i] for i in self.polarity]

    def set_polarity(self, y:List[int]):
        self.polarity = [int2polarity[i] for i in y]

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.__dict__)

class ACSAItems(object): # 用于ATSA任务的项
    def __init__(self, data:List[tuple]):
        self.id = []
        self.text = []
        self.term = []
        self.polarity = []
        self.size = len(data)
        for i in data:
            self.id.append(i[0])
            self.text.append(i[1])
            self.term.append(i[2])
            self.polarity.append(i[3])

    def __getitem__(self, idx):
        return self.text[idx], self.term[idx], self.polarity[idx]
    
    def to_instances(self):
        instances = []
        for i in range(self.size):
            if len(instances) == 0 or self.id[i] != instances[-1].id: # 需要添加新的instance
                instances.append(Instancev2(self.id[i], self.text[i]))
            instances[-1].add_aspect_category(self.term[i], self.polarity[i])
        return instances

    def get_y(self):
        return [polarity2int[i] for i in self.polarity]

    def set_polarity(self, y:List[int]):
        self.polarity = [int2polarity[i] for i in y]

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.__dict__)


class Semeval2014DataSet(object):
    def __init__(self, val:Union[str, ATSAItems, ACSAItems]):
        self.instances = None
        self.size = 0
        self.terms_statistics = None
        self.categories_statistics = None

        if isinstance(val, str): # 从文件中加载
            self._from_xml(val)
        elif isinstance(val, ATSAItems):
            self._from_instances(val.to_instances())
        elif isinstance(val, ACSAItems):
            self._from_instances(val.to_instances())
        
    def _from_instances(self, instances:List[sb.Instance or Instancev2]) -> NoReturn:
        self.instances = instances
        self.size = len(self.instances)  
        self._update_aspects_terms_statistics()
        self._update_aspects_categories_statistics()
        
    def _from_xml(self, filename:str) -> NoReturn:
        sentences = ET.parse(filename).getroot().findall("sentence")
        self._from_instances([sb.Instance(i) for i in sentences])

    def _update_aspects_terms_statistics(self) -> NoReturn: #更新目标词信息
        num_statistics = {}
        term_statistics = {}
        for i in self.instances:
            for at in i.aspect_terms:
                term_statistics[at.term] = term_statistics.get(at.term, 0) + 1
                num_statistics[at.polarity] = num_statistics.get(at.polarity, 0) + 1
        self.terms_statistics = (term_statistics, num_statistics) 

    def _update_aspects_categories_statistics(self) -> NoReturn: # 更新方面词信息
        num_statistics = {}
        category_statistics = {}

        for i in self.instances:
            for ac in i.aspect_categories:
                category_statistics[ac.term] = category_statistics.get(ac.term, 0) + 1
                num_statistics[ac.polarity] = num_statistics.get(ac.polarity, 0) + 1
        self.categories_statistics = (category_statistics, num_statistics) 

    def get_ATSAItems(self, clear:bool=True) -> ATSAItems: # 获取用于ATSA任务的数据
        res = []
        for i in self.instances:
            for at in i.aspect_terms: # 同一id的多个terms一定相邻
                if clear and at.polarity != "conflict": # 排除掉conflict标签
                    res.append((i.id, i.text, at.term, at.polarity))
        return ATSAItems(res)
    

    def get_ACSAItems(self, clear:bool=True) -> ACSAItems: # 获取用于ACSA任务的数据
        res = []
        for i in self.instances:
            for at in i.aspect_categories: # 同一id的多个terms一定相邻
                if clear and at.polarity != "conflict": # 排除掉conflict标签
                    res.append((i.id, i.text, at.term, at.polarity))
        return ACSAItems(res)
    
    def dump(self, filename:str, has_lable=True): # 默认保存aspect terms 和 aspect categories
        c = Corpusv2()
        c.write_out(filename, self.instances, not has_lable) # 写入结果到文件
    


def classifer_evaluate(y_true, y_pred) -> Tuple[float, float]:
    return metrics.accuracy_score(y_true, y_pred), metrics.f1_score(y_true, y_pred, average="macro")


def main():
    pass

if __name__ == "__main__":
    main()