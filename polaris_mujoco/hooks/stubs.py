"""Import-time hook: serve stub modules for isaaclab*/isaacsim*/omni*/diff_surfel_rasterization
so upstream `polaris` imports unmodified. Stubs RECORD constructor kwargs as attributes."""
import sys, types, importlib.abc, importlib.machinery

class StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith('__'): raise AttributeError(name)
        return type(name, (Stub,), {})

class Stub(metaclass=StubMeta):
    def __init__(self, *args, **kw):
        self.args = args
        self.__dict__.update(kw)
    def __call__(self, *a, **k):  # decorators like @configclass, or FRAME_MARKER_CFG.copy()
        return a[0] if len(a) == 1 and not k and callable(a[0]) else Stub(*a, **k)
    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        return Stub()
    def copy(self): return self
    def __getitem__(self, k): return Stub()
    def __repr__(self): return f"Stub({ {k:v for k,v in self.__dict__.items() if k!='args'} })"

def configclass(cls=None, **kw):
    return cls if cls is not None else (lambda c: c)

class StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        if name == 'configclass': return configclass
        # anything else: a class-like stub (subclassable, instantiable)
        return type(name, (Stub,), {})

PREFIXES = ('isaaclab', 'isaacsim', 'omni', 'isaaclab_tasks')   # NOT the CUDA kernels: those are real here

class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in PREFIXES:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        m = StubModule(spec.name); m.__path__ = []; return m
    def exec_module(self, module): pass

def install():
    sys.meta_path.insert(0, Finder())
    # real pxr lacks Isaac's Semantics schema -> attach a stub attribute (hook, not edit)
    import pxr; pxr.Semantics = type('Semantics', (), {'SemanticsAPI': Stub})
    # omni.usd.get_context().get_stage() -> we hand the stage in from outside
    import omni.usd
    omni.usd.get_context = lambda: Stub(get_stage=lambda: STAGE['stage'])

STAGE = {'stage': None}
