import re,os,glob
loop_flags = ['-floop-interchange', '-floop-unroll-and-jam', '-fpeel-loops', '-fsplit-loops',
              '-funswitch-loops', '-fmove-loop-invariants', '-ftree-loop-distribute-patterns',
              '-ftree-loop-distribution', '-ftree-loop-im', '-ftree-loop-optimize', '-ftree-loop-vectorize',
              '-fversion-loops-for-strides', '-fsplit-paths']
branch_flags = ['-fthread-jumps', '-fif-conversion', '-fif-conversion2', '-fhoist-adjacent-loads',
                  '-fprintf-return-value', '-ftree-tail-merge', '-fguess-branch-probability', '-fcse-follow-jumps',
                  '-ftree-dominator-opts', '-freorder-blocks', '-freorder-blocks-and-partition','-ftree-ch']
function_flags = ['-fipa-sra', '-ftree-pta', '-ftree-builtin-call-dce', '-fshrink-wrap',
                  '-freorder-functions', '-fcaller-saves', '-fdefer-pop', '-fdevirtualize',
                  '-fdevirtualize-speculatively', '-ffunction-cse', '-findirect-inlining',
                  '-finline-functions', '-finline-functions-called-once', '-finline-small-functions',
                  '-fipa-cp-clone', '-fipa-icf-functions', '-fipa-modref', '-fipa-profile', '-fipa-pure-const',
                  '-fipa-ra', '-fpartial-inlining']
static_variable_flags = ['-fipa-reference', '-fipa-reference-addressable', '-ftoplevel-reorder', 
                         '-fipa-icf-variables', '-ftree-bit-ccp', '-ftree-ccp', '-ftree-coalesce-vars']
pointer_flags = ['-fisolate-erroneous-paths-dereference', '-fomit-frame-pointer', '-ftree-vrp']
string_flags = ['-foptimize-strlen']
float_flags = ['-fsigned-zeros']

all_flags = ['-falign-functions', '-falign-jumps', '-falign-labels', '-falign-loops', '-fasynchronous-unwind-tables', '-fbit-tests', 
             '-fbranch-count-reg', '-fcaller-saves', '-fcode-hoisting', '-fcombine-stack-adjustments', '-fcompare-elim', '-fcprop-registers',
             '-fcrossjumping', '-fcse-follow-jumps', '-fdefer-pop', '-fdevirtualize', '-fdevirtualize-speculatively', '-fdse',
             '-fexpensive-optimizations', '-fforward-propagate', '-ffunction-cse', '-fgcse', '-fgcse-after-reload', 
             '-fguess-branch-probability', '-fhoist-adjacent-loads', '-fif-conversion', '-fif-conversion2', '-findirect-inlining', '-finline', 
             '-finline-functions', '-finline-functions-called-once', '-finline-small-functions', '-fipa-bit-cp', 
             '-fipa-cp', '-fipa-cp-clone', '-fipa-icf', '-fipa-icf-functions', '-fipa-icf-variables', '-fipa-modref',
             '-fipa-profile', '-fipa-pure-const', '-fipa-ra', '-fipa-reference', '-fipa-reference-addressable', 
             '-fipa-sra', '-fipa-vrp', '-fira-share-save-slots', '-fisolate-erroneous-paths-dereference', '-fjump-tables', 
             '-floop-interchange', '-floop-unroll-and-jam', '-flra-remat', '-fmove-loop-invariants', '-fomit-frame-pointer', 
             '-foptimize-sibling-calls', '-foptimize-strlen', '-fpartial-inlining', '-fpeel-loops', '-fpeephole2', 
             '-fpredictive-commoning', '-fprintf-return-value', '-free', '-frename-registers', '-freorder-blocks',
            '-freorder-blocks-and-partition', '-freorder-functions', '-frerun-cse-after-loop', '-fsched-dep-count-heuristic', 
            '-fsched-interblock', '-fsched-rank-heuristic', '-fsched-spec-insn-heuristic', '-fschedule-fusion',
            '-fschedule-insns2', '-fshrink-wrap', '-fsigned-zeros', '-fsplit-loops', '-fsplit-paths', 
            '-fsplit-wide-types', '-fssa-phiopt', '-fstore-merging', '-fstrict-aliasing', '-fthread-jumps', 
            '-ftoplevel-reorder', '-ftree-bit-ccp', '-ftree-builtin-call-dce', '-ftree-ccp', '-ftree-ch', 
            '-ftree-coalesce-vars', '-ftree-copy-prop', '-ftree-dce', '-ftree-dominator-opts', '-ftree-dse', 
            '-ftree-fre', '-ftree-loop-distribute-patterns', '-ftree-loop-distribution', 
            '-ftree-loop-im', '-ftree-loop-optimize', '-ftree-loop-vectorize', '-ftree-partial-pre', 
            '-ftree-pre', '-ftree-pta', '-ftree-scev-cprop', '-ftree-sink', '-ftree-slp-vectorize', '-ftree-slsr', 
            '-ftree-sra', '-ftree-switch-conversion', '-ftree-tail-merge', 
            '-ftree-ter', '-ftree-vrp', '-funroll-completely-grow-size', '-funswitch-loops', '-fvar-tracking', '-fversion-loops-for-strides']


def obtain_c_code(file_path):
    """
    param file_path: source code folder
    return: .c code (without loop-wrap.c)
    """
    c_code = ""
    # find *.c
    pattern = os.path.join(file_path, '*.c')
    for file_path in glob.glob(pattern):
        # obtain file name
        filename = os.path.basename(file_path)
        if filename != 'loop-wrap.c':
            with open(file_path, 'r') as file:
                c_code += file.read() + "\n"  
    return c_code

def contain_loop(code):
    for_loop_pattern = r"for\s*\(([^)]+)\)\s*{?"
    while_loop_pattern = r"while\s*\(([^)]+)\)\s*{?"
    do_while_loop_pattern = r"do\s*{?[^}]*}\s*while\s*\(([^)]+)\)"
    for_loops = re.findall(for_loop_pattern, code)
    while_loops = re.findall(while_loop_pattern, code)
    do_while_loops = re.findall(do_while_loop_pattern, code)
    matches = for_loops + while_loops + do_while_loops
    matches = list(set(matches))
    if len(matches) > 0:
        return True
    else:
        return False

def contain_branch(code):
    """
    branch
    """
    pattern = r'if\s*\(.*?\)|else\s*if\s*\(.*?\)|else|switch\s*\(.*?\)'
    matches = re.findall(pattern, code)
    matches = list(set(matches))
    if len(matches) > 0:
        return True
    else:
        return False

def contain_function(code):
    """
    call and declaration
    """
    function_call_pattern = r'\b\w+\s*\([^)]*\)'
    function_declaration_pattern = r'\b\w+\s+\w+\s*\([^)]*\)'
    function_calls = re.findall(function_call_pattern, code)
    function_declarations = re.findall(function_declaration_pattern, code)
    function_calls_names = set([re.match(r'\b\w+', call).group() for call in function_calls])
    function_declarations_names = set([re.match(r'\b\w+\s+(\w+)', decl).group(1) for decl in function_declarations])
    matched_functions = function_calls_names.intersection(function_declarations_names)
    if 'main' in matched_functions:
        matched_functions.remove('main')
    if 'int' in matched_functions:
        matched_functions.remove('int')
    if 'float' in matched_functions:
        matched_functions.remove('float')
    if 'double' in matched_functions:
        matched_functions.remove('double')
    if 'string' in matched_functions:
        matched_functions.remove('string')
    if 'long' in matched_functions:
        matched_functions.remove('long')
    return list(matched_functions)

def contain_static_variable(code):
    """
    static variable
    """
    pattern = r'\bstatic\s+\w+\s+\w+\s*=?\s*[^;]*'
    matches = re.findall(pattern, code)
    if len(matches) > 0:
        return True
    else:
        return False

def contain_pointer(code):
    """
    pointer
    """
    pattern = r'\b([_a-zA-Z][_a-zA-Z0-9]*\s+\*+\s*[_a-zA-Z][_a-zA-Z0-9]*\s*);'
    # pattern = r"\b\w+\s+\*\w+;|\b\w+\s*=\s*(&\w+|\w+);|\*\w+\s*=\s*\d+;|\b\w+->\w+\s*=\s*\w+;"
    matches = re.findall(pattern, code)
    matches = list(set(matches))
    if len(matches) > 0:
        return True
    else:
        return False

def contain_string(code):
    """
    string function
    """
    pattern = r'\b(str(?:len|cpy|ncpy|cat|ncat|cmp|ncmp|chr|rchr|str|tok|dup|ncpy))\b'
    matches = re.findall(pattern, code)
    if len(matches) > 0:
        return True
    else:
        return False

def contain_float_calculation(code):
    float_pattern = r'[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?'
    matches = re.findall(float_pattern, code)

def remove_commentsandinclude_from_c_code(c_code):
    """
    obtian pure code
    """
    # /**/
    c_code = re.sub(r'/\*.*?\*/', '', c_code, flags=re.DOTALL)
    # //
    c_code = re.sub(r'//.*', '', c_code)
    # /*
    c_code = re.sub(r'".*?"', lambda x: x.group(0) if '/*' not in x.group(0) else '', c_code)
    # include
    c_code = re.sub(r'#include\s*<.*?>', '', c_code)
    c_code = re.sub(r'#include\s*".*?"', '', c_code)
    return "\n".join([line for line in c_code.split('\n') if line.strip() != ''])


def get_related_flags(code):
    """
    obtain the potential related flags
    """
    related_idx = []
    related_flags = []
    code_without_comment = remove_commentsandinclude_from_c_code(code)
    if contain_loop(code_without_comment):
        related_flags = related_flags + loop_flags
    if contain_branch(code_without_comment):
        related_flags = related_flags + branch_flags
    if contain_function(code_without_comment):
        related_flags = related_flags + function_flags
    if contain_static_variable(code_without_comment):
        related_flags = related_flags + static_variable_flags
    if contain_pointer(code_without_comment):
        related_flags = related_flags + pointer_flags
    if contain_string(code_without_comment):
        related_flags = related_flags + string_flags
    if contain_float_calculation(code_without_comment):
        related_flags = related_flags + float_flags
    
    for flag in related_flags:
        position = all_flags.index(flag)
        related_idx.append(position)
        
    return sorted(related_idx)

if __name__ == "__main__":
    code = obtain_c_code('/data/zmx/CFSCA/cBench/automotive_susan_s/src')
    new_code = remove_commentsandinclude_from_c_code(code)
    print(get_related_flags(new_code))