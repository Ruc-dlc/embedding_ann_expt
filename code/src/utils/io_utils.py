"""
文件IO工具

本模块提供常用的文件读写功能。
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

def save_json(
    data: Union[Dict, List],
    path: str,
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    保存JSON文件
    
    Args:
        data: 要保存的数据
        path: 文件路径
        indent: 缩进空格数
        ensure_ascii: 是否确保ASCII编码
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

def load_json(path: str) -> Union[Dict, List]:
    """
    加载JSON文件
    
    Args:
        path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(data: Any, path: str) -> None:
    """
    保存Pickle文件
    
    Args:
        data: 要保存的数据
        path: 文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path: str) -> Any:
    """
    加载Pickle文件
    
    Args:
        path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_embeddings(
    embeddings,
    path: str,
    ids: Optional[List[str]] = None
) -> None:
    """
    保存向量文件
    
    Args:
        embeddings: numpy数组或torch张量
        path: 文件路径
        ids: 向量对应的ID列表
    """
    import numpy as np
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为numpy
    if hasattr(embeddings, 'cpu'):
        embeddings = embeddings.cpu().numpy()
        
    np.save(path, embeddings)
    
    # 保存ID映射
    if ids is not None:
        id_path = path.with_suffix('.ids.txt')
        with open(id_path, 'w') as f:
            for id_ in ids:
                f.write(f"{id_}\n")

def load_embeddings(path: str, load_ids: bool = False):
    """
    加载向量文件
    
    Args:
        path: 文件路径
        load_ids: 是否加载ID映射
        
    Returns:
        向量数组，或 (向量数组, ID列表)
    """
    import numpy as np
    
    path = Path(path)
    embeddings = np.load(path)
    
    if load_ids:
        id_path = path.with_suffix('.ids.txt')
        if id_path.exists():
            with open(id_path, 'r') as f:
                ids = [line.strip() for line in f]
            return embeddings, ids
        return embeddings, None
        
    return embeddings

def save_results(
    results: Dict[str, Any],
    path: str,
    format: str = "json"
) -> None:
    """
    保存实验结果
    
    Args:
        results: 结果字典
        path: 文件路径
        format: 保存格式 ("json", "csv", "yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        save_json(results, str(path))
    elif format == "csv":
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
    elif format == "yaml":
        import yaml
        with open(path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_tsv(path: str, has_header: bool = True) -> List[Dict[str, str]]:
    """
    加载TSV文件
    
    Args:
        path: 文件路径
        has_header: 是否有表头
        
    Returns:
        记录列表
    """
    import csv
    
    with open(path, 'r', encoding='utf-8') as f:
        if has_header:
            reader = csv.DictReader(f, delimiter='\t')
            return list(reader)
        else:
            reader = csv.reader(f, delimiter='\t')
            return [row for row in reader]

def ensure_dir(path: str) -> Path:
    """
    确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path