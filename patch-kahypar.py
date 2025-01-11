import re
from pathlib import Path

pat = re.compile(r'LOG << (?P<err>.*)\n(?P<ident>\s*)(?P<exit>(?:std::)?exit.*)')

def replace_exit(text: str) -> str:
    def sub_fn(mat: re.Match):
        err = mat.group('err')
        ident = mat.group('ident')
        return f'{ident}throw std::runtime_error(({{std::stringstream ss; ss << {err} ss.str();}}));'
    return pat.sub(sub_fn, text)

for file in Path('kahypar').glob('**/*.h'):
    text = file.read_text()
    text_rep = replace_exit(text)
    if text_rep != text:
        print(f'Update {file}')
        text_rep = text_rep.replace('#pragma once', '#pragma once\n#include<sstream>')
        file.write_text(text_rep)

file = Path('external_tools/kahypar-shared-resources/kahypar-resources/macros.h')
text = file.read_text()
repl = '''\
#define ALWAYS_ASSERT_2(cond, msg)            \\
  do {                                        \\
    if (!(cond)) {                            \\
      DBG1 << "Assertion `" #cond "` failed:" \\
           << msg;                            \\
      std::abort();                           \\
    }                                         \\
  } while (0)
'''
tgt = '''\
#define ALWAYS_ASSERT_2(cond, msg)            \\
    do { throw std::runtime_error(({std::stringstream ss; ss << "Assertion `" #cond "` failed:" << msg; ss.str();})); } while(0)
'''
text = text.replace(repl, tgt)
text = text.replace('#pragma once', '#pragma once\n#include<sstream>')
file.write_text(text)