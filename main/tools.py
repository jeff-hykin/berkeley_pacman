def create_named_list_class(names):
    """
    Example:
        Position = create_named_list_class(['x','y','z'])
        a = Position([1,2,3])
        print(a.x)   # 1
        a.x = 4
        print(a[0])  # 4
        a[0] = 9
        print(a.x)   # 9
    """
    
    names_to_index = {}
    if isinstance(names, dict):
        names_to_index = names
    if isinstance(names, (tuple, list)):
        for index, each in enumerate(names):
            names_to_index[each] = index
    
    class NamedList(list):
        def __getitem__(self, key):
            if isinstance(key, (int, slice)):
                return super(NamedList, self).__getitem__(key)
            # assume its a name
            else:
                try:
                    index = names_to_index[key]
                except:
                    raise KeyError(f'''key={key} not in named list: {self}''')
                if index >= len(self):
                    return None
                return self[index]
        
        def __getattr__(self, key):
            if key in names_to_index:
                return self[key]
            else:
                super(NamedList, self).__getattr__(key)
        
        def __setattr__(self, key, value):
            if key in names_to_index:
                index = names_to_index[key]
                while index >= len(self):
                    super(NamedList, self).append(None)
                super(NamedList, self).__setitem__(index, value)
            else:
                super(NamedList, self).__setattr__(key, value)
        
        def __setitem__(self, key, value):
            if isinstance(key, int):
                super(NamedList, self).__setitem__(key, value)
            # assume its a name
            else:
                index = names_to_index[key]
                while index >= len(self):
                    super(NamedList, self).append(None)
                super(NamedList, self).__setitem__(index, value)
                
        def keys(self):
            return list(names_to_index.keys())
        
        def values(self):
            return self
        
        def get(self, key, default):
            try:
                return self[key]
            except Exception as error:
                return default
        
        def items(self):
            return zip(self.keys(), self.values())
        
        def update(self, other):
            for each_key in names_to_index:
                if each_key in other:
                    self[each_key] = other[each_key]
            return self
        
        def __repr__(self):
            import itertools
            out_string = '['
            named_values = 0
            
            reverse_lookup = {}
            for each_name, each_index in names_to_index.items():
                reverse_lookup[each_index] = reverse_lookup.get(each_index, []) + [ each_name ]
                
            for each_index, value in enumerate(self):
                name = "=".join(reverse_lookup.get(each_index, []))
                if name:
                    name += '='
                out_string += f' {name}{value},'
            
            out_string += ' ]'
            return out_string
            
    return NamedList


Transition = create_named_list_class(['state', 'action', 'cost'])