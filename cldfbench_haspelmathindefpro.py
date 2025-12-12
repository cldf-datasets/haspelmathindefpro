import pathlib
import re
import unicodedata
from collections import defaultdict, namedtuple
from itertools import chain, islice, zip_longest

from cldfbench import Dataset as BaseDataset, CLDFSpec


IDCHARS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789-_')


def slug(s):
    return ''.join(
        c.lower()
        for c in unicodedata.normalize('NFKD', s)
        if c in IDCHARS)


def make_parameters(csv_rows):
    return {row['Original_Name']: row for row in csv_rows}


TxtExample = namedtuple('TxtExample', 'glottocode feature lines')


def extract_examples(txt_lines):
    lines = iter(txt_lines)
    glottocode = None
    feature = None
    example = []
    while True:
        try:
            line = next(lines).lstrip(' ').rstrip()
        except StopIteration:
            break
        if re.fullmatch(r'[a-z]{4}\d{4}', line):
            glottocode = line
            # consume language name
            _ = next(lines)
        elif re.match(r'\S', line):
            feature = line
            if example:
                assert glottocode
                assert feature
                yield TxtExample(
                    glottocode=glottocode,
                    feature=feature,
                    lines=example)
            example = []
        elif re.match(r'\t', line):
            example.append(line)
        elif not line.strip():
            if example:
                assert glottocode
                assert feature
                yield TxtExample(
                    glottocode=glottocode,
                    feature=feature,
                    lines=example)
            example = []
        else:
            raise AssertionError(f'unexpected line: "{line}"')
    if example:
        assert glottocode
        assert feature
        yield TxtExample(
            glottocode=glottocode,
            feature=feature,
            lines=example)


class ExampleIdMaker:
    def __init__(self):
        self.seen = {}

    def make_id(self, language_id):
        count = self.seen.get(language_id) or 0
        self.seen[language_id] = count + 1
        return f'{language_id}-{count + 1}'


def rescue_unicode_chars(word, glottocode):
    if glottocode == 'icel1247':
        # icelandic
        word = re.sub('([^ ‘])D', r'\1ð', word)
        # we were lucky that all thorns were at the beginning of words and all
        # p's somewhere in the middle (<_<)"
        word = re.sub('^P', 'Þ', word)
        word = re.sub('^p', 'þ', word)
    elif glottocode == 'roma1327':
        # romanian
        word = word.replace('a*', 'ă')
        word = word.replace('s3', 'ș')
        word = word.replace('t3', 'ț')
        # not a haček but a breve
        word = word.replace(r'a;\D\fo1(', 'ă')
    elif glottocode == 'poli1260':
        # polish
        word = word.replace('e4', 'ę')
        word = word.replace('a4', 'ą')
        word = word.replace('z5', 'ż')
        word = word.replace('z%', 'ż')
        # not a haček but a kropka
        word = word.replace(r';\D\fo1(', chr(0x0301))
    elif glottocode == 'sout1528':
        # serbo-croatian
        word = word.replace(r'istuc;\D\fo1(i', 'istući')
    elif glottocode == 'lith1251':
        # lithuanian
        word = re.sub('^ka4?$', 'ką', word)
        word = word.replace('u4', 'ų')
        word = word.replace('e%', 'ė')
        word = re.sub(r'^s;\\D\\fo1\(ia$', 'šią', word)
        word = re.sub('^problema$', 'problemą', word)
        word = re.sub('^nauju$', 'naujų', word)
        word = re.sub('^pusiu$', 'pusių', word)
        word = re.sub(r'^jokiu(/?)$', r'jokių\1', word)
        word = re.sub('^meteoru$', 'meteorų', word)
        word = re.sub('^nakti$', 'naktį', word)
        word = re.sub('^desningumu$', 'desningumų', word)
    elif glottocode == 'latv1249':
        # latvian
        # not a haček but a macron
        word = re.sub(r'([aeiou]);\\D\\fo1\(', f'\\1{chr(0x0304)}', word)
        word = re.sub(r'^([vV])ins;\\D\\fo1\(', r'\1iņš', word)
        word = re.sub(r'^([vV])inu', r'\1iņu', word)
        word = re.sub('^pārmainām$', r'pārmaiņām', word)
        word = word.replace('a@', 'ā')
    elif glottocode == 'west2369':
        # persian
        word = re.sub(r'(\w)\?(\w)', r'\1ʔ\2', word)
    elif glottocode == 'hind1269':
        # hindi
        # not a haček but a tilde
        word = re.sub(r'([aeiou]);\\D\\fo1\(', f'\\1{chr(0x0303)}', word)
        word = word.replace('lark', 'laṛk')
    elif glottocode == 'nucl1301':
        # turkish
        # not a haček but a breve
        word = word.replace(r'g;\D\fo1(', 'ğ')
        word = word.replace('s3', 'ş')
    elif glottocode in {'kaza1248', 'yaku1245'}:
        # kazakh and yakut
        word = word.replace('V', 'ɣ')
        word = re.sub(r'([a-z\-])N', r'\1ŋ', word)
    elif glottocode == 'lezg1247':
        # lezgian
        # not a haček but a circumflex
        word = word.replace(r'x;\D\fo1(', 'x̂')
        word = word.replace('qh', 'qʰ')
    elif glottocode == 'malt1254':
        # check maltese
        # not a haček but a dot above
        word = word.replace(r';\S\up4(\D\fo2(', chr(0x0307))
        word = word.replace('Ì', 'ħ')
    elif glottocode == 'hebr1245':
        # hebrew
        word = re.sub(r'(\w)\?(\w)', r'\1ʔ\2', word)
        word = word.replace('Ì', 'ħ')
    elif glottocode == 'nucl1302':
        # georgian
        if word != 'Vera-prit':
            word = word.replace('V', 'ɣ')
        word = word.replace('J', 'ʒ')
    elif glottocode == 'nucl1305':
        # kannada
        word = word.replace(r's;\D\fo1(', 'ś')
        word = word.replace('d5', 'ḍ')
        word = word.replace('l5', 'ḷ')
    elif glottocode == 'mand1415':
        # not haček but macron
        word = re.sub(r'^Chi;\\D\\fo1\($', 'Chī', word)
        word = re.sub(r'^([tT])a;\\D\\fo1\($', r'\1ā', word)
        word = re.sub(r'^zhi;\\D\\fo1\(dao$', 'zhīdao', word)
        word = re.sub(r'^yi;\\D\\fo1\($', 'yī', word)
        word = re.sub(r'^([cC])a;\\D\\fo1\(i$', r'\1āi', word)
        word = re.sub(r'do;\\D\\fo1\(u', 'dōu', word)
        word = re.sub(r'^co;\\D\\fo1\(ngmíng\.?$', 'cōngmíng', word)
        word = re.sub(r'^yi;\\D\\fo1\(ge$', 'yīge', word)
        word = re.sub(r'^nánshe;\\D\\fo1\(ng$', 'nánshēng', word)

    # haček (hopefully)
    word = word.replace(r'i;\D\fo1(', 'ǐ')
    word = word.replace(r';\D\fo1(', chr(0x030C))
    word = word.replace(r';\S\up2(\D\fo1(', chr(0x030C))
    word = word.replace(r';\S\up2(\D\fo2(', chr(0x030C))

    # get rid of the combining diacritics
    return unicodedata.normalize('NFC', word)


def normalise_example_text(text, glottocode):
    text = text.strip()
    words = text.split()
    if words and words[0] == 'b.':
        words = words[1:]
    text = ' '.join(rescue_unicode_chars(w, glottocode) for w in words)
    text = text.replace('’', "'").replace('‘', "'")
    return text


def strip_quotes(text):
    return text.strip().lstrip("‘'").rstrip("’'")


def extract_example_source(txt_example):
    lines = txt_example.lines
    assert lines
    source = None
    if lines[-1][-1] == ')':
        lastline = lines[-1]
        source_start = lastline.index('(')
        # ) <= vim is stupid sometimes
        assert source_start >= 0, lines
        source = lastline[source_start:]
        rest = lastline[:source_start].strip()
        new_lines = lines[:-1]
        if rest:
            new_lines.append(rest)
        txt_example = txt_example._replace(lines=new_lines)
    return source, txt_example


def merge_translation_lines(txt_example):
    lines = txt_example.lines
    assert lines
    assert lines[-1][-1] in "’'", lines
    translation_start = None
    for i, line in enumerate(lines):
        if line.strip()[0] in "‘'":
            translation_start = i
    assert translation_start is not None, lines
    translation = ' '.join(islice(lines, translation_start, None))
    new_lines = [*islice(lines, translation_start), translation]
    return txt_example._replace(lines=new_lines)


def fix_translation(txt_example):
    fixes = {
        '\t\t‘Nobody saw anything.’ (Or: ‘Someone saw nothing.’)':
        '\t\t‘Nobody saw anything. (Or: Someone saw nothing.)’',
        '\t‘Call somewhere (or other)!':
        '\t‘Call somewhere (or other)!’',
        "\tThe weather in San Sebastián is more pleasant than anywhere else.'":
        '\t‘The weather in San Sebastián is more pleasant than anywhere else.’',
    }
    if (replacement := fixes.get(txt_example.lines[-1])):
        return txt_example._replace(
            lines=[*txt_example.lines[:-1], replacement])
    else:
        return txt_example


def make_example(txt_example, id_maker):
    _source, txt_example = extract_example_source(txt_example)
    txt_example = fix_translation(txt_example)
    if re.match(r'\t*\(', txt_example.lines[-1]):
        # ) <= vim is stupid sometimes
        txt_example = txt_example._replace(
            lines=txt_example.lines[:-1])
    if 'will:tell' in txt_example.lines[-1]:
        # FIXME: remove this hack when the example was fixed
        assert len(txt_example.lines) == 2, 'remove this hack when the example was fixed'
        txt_example = txt_example._replace(lines=[*txt_example.lines, '‘’'])
    if txt_example.lines[-1] == '\t‘Someone called. I don’t know who.’':
        lines = txt_example.lines
        assert len(lines) == 5
        new_lines = [
            '\t'.join((lines[0], lines[2].lstrip('\t'))),
            '\t'.join((lines[1], lines[3].lstrip('\t'))),
            lines[4],
        ]
        txt_example = txt_example._replace(lines=new_lines)

    txt_example = merge_translation_lines(txt_example)
    gc = txt_example.glottocode

    if len(txt_example.lines) == 2:
        return {
            'ID': id_maker.make_id(gc),
            'Language_ID': gc,
            'Primary_Text': normalise_example_text(txt_example.lines[0], gc),
            'Translated_Text': normalise_example_text(strip_quotes(txt_example.lines[1]), gc),
            'Parameter_ID': txt_example.feature,
        }
    elif len(txt_example.lines) == 3:
        return {
            'ID': id_maker.make_id(gc),
            'Language_ID': gc,
            'Primary_Text': normalise_example_text(txt_example.lines[0], gc),
            'Analyzed_Word': [
                normalise_example_text(word, gc)
                for word in txt_example.lines[0].lstrip('\t').split('\t')],
            'Gloss': [
                normalise_example_text(word, gc)
                for word in txt_example.lines[1].lstrip('\t').split('\t')],
            'Translated_Text': normalise_example_text(strip_quotes(txt_example.lines[2]), gc),
            'Parameter_ID': txt_example.feature,
        }
    else:
        raise AssertionError(txt_example.lines)


def make_examples(txt_examples):
    id_maker = ExampleIdMaker()
    return [
        example
        for txt_example in txt_examples
        if (example := make_example(txt_example, id_maker))]


def visual_len(s):
    return sum(1 for c in s if unicodedata.category(c) not in {'Mn', 'Me', 'Cf'})


def visual_pad(s, new_width):
    vl = visual_len(s)
    return '{}{}'.format(s, ' ' * (new_width - vl)) if new_width > vl else s


def aligned_example(analysed, gloss, indent=0):
    widths = [
        max(visual_len(a), visual_len(g))
        for a, g in zip_longest(analysed, gloss, fillvalue='')]
    prefix = ' ' * indent if indent else ''
    line1 = '  '.join(visual_pad(a, w) for a, w in zip(analysed, widths))
    line2 = '  '.join(visual_pad(g, w) for g, w in zip(gloss, widths))
    return f'{prefix}{line1}\n{prefix}{line2}'


def glosses_are_aligned(example, languages):
    analysed = example.get('Analyzed_Word') or []
    gloss = example.get('Gloss') or []
    if len(analysed) == len(gloss):
        return True
    else:
        example_id = example['ID']
        primary = example['Primary_Text']
        translation = example['Translated_Text']
        language = languages[example['Language_ID']]['Name']
        print(f'example {example_id} ({language}): ERR: misaligned gloss')
        print(' ', primary)
        print(aligned_example(analysed, gloss, indent=2))
        print(f'  ‘{translation}’')
        print()
        return False


def make_languages(data, glottolog):
    gc2name = {row['Glottocode']: row['language'] for row in data}
    languoids = sorted(
        glottolog.languoids(ids=gc2name),
        key=lambda lg: lg.id)
    return [
        {
            'ID': lg.id,
            'Glottocode': lg.id,
            'Name': gc2name[lg.id],
            'ISO639P3code': lg.iso,
            'Latitude': lg.latitude,
            'Longitude': lg.longitude,
            'Macroarea': lg.macroareas[0].name if lg.macroareas else '',
        }
        for lg in languoids]


def make_constructions(data):
    return {
        (row['Glottocode'], row['form']): {
            'ID': '{}-{}'.format(row['Glottocode'], slug(row['form'])),
            'Name': 'Marker: {}'.format(row['form']),
            'Language_ID': row['Glottocode']}
        for row in data}


def _iter_ccodes(cparamaters):
    for param in cparamaters.values():
        param_id = param['ID']
        yield {
            'ID': f'{param_id}-no',
            'Parameter_ID': param_id,
            'Name': 'no'}
        yield {
            'ID': f'{param_id}-yes',
            'Parameter_ID': param_id,
            'Name': 'yes'}


def make_ccodes(cparameters):
    return list(_iter_ccodes(cparameters))


def make_cvalue(row, constructions, cparameter):
    glottocode = row['Glottocode']
    form = row['form']
    construction = constructions[glottocode, form]
    value_id = '{}-{}'.format(construction['ID'], cparameter['ID'])
    cell = row[cparameter['Original_Name']]
    if cell == '0':
        value = 'no'
    elif cell == '1':
        value = 'yes'
    else:
        raise ValueError(f'{value_id}: value must be 0 or 1, got {cell}')
    return {
        'ID': value_id,
        'Construction_ID': construction['ID'],
        'Parameter_ID': cparameter['ID'],
        'Code_ID': '{}-{}'.format(cparameter['ID'], value),
        'Value': value,
    }


def make_cvalues(data, constructions, cparameters):
    return [
        make_cvalue(row, constructions, param)
        for row in data
        for colname, param in cparameters.items()]


def make_lvalues(data, lparameters, examples):
    feature_id_fixes = {
        'irrealis non-specific': ['irrealis nsp'],
        'direct negatio': ['direct negation'],
        'free-choice': ['free choice'],
        'irrealis non-specific (imperative)': ['irrealis nsp'],
        'irrealis non-specific (‘want’)': ['irrealis nsp'],
        'irrealis-non-specific': ['irrealis nsp'],
        'specific': ['specific unknown', 'specific known'],
        'specific known/unknown': ['specific unknown', 'specific known'],
        'unknown': ['specific unknown'],
    }
    param_examples = defaultdict(list)
    for ex in examples:
        colname = ex['Parameter_ID']
        glottocode = ex['Language_ID']
        colnames = feature_id_fixes.get(colname) or [colname]
        for colname in colnames:
            parameter_id = lparameters[colname]['ID']
            param_examples[glottocode, parameter_id].append(ex['ID'])

    forms = defaultdict(list)
    for row in data:
        glottocode = row['Glottocode']
        form = row['form']
        for colname in row:
            if (param := lparameters.get(colname)):
                forms[glottocode, param['ID']].append(form)
    return [
        {
            'ID': f'{glottocode}-{param_id}',
            'Language_ID': glottocode,
            'Parameter_ID': param_id,
            'Value': ' / '.join(param_forms),
            'Example_IDs': param_examples.get((glottocode, param_id)) or [],
        }
        for (glottocode, param_id), param_forms in forms.items()]


def add_construction_descriptions(constructions, cvalues, parameters):
    param_meanings = {
        param['ID']: re.fullmatch('Expresses (‘[^’]+’)', param['Name']).group(1)
        for param in parameters.values()}

    construction_meanings = defaultdict(list)
    for val in cvalues:
        if val['Value'] == 'yes':
            meaning = param_meanings[val['Parameter_ID']]
            construction_meanings[val['Construction_ID']].append(meaning)

    for construction in constructions.values():
        meanings = '; '.join(construction_meanings[construction['ID']])
        construction['Description'] = f'Meaning: {meanings}'


def make_schema(cldf):
    cldf.add_component('LanguageTable')
    cldf.add_component('ParameterTable')
    cldf.add_component('CodeTable')
    cldf.add_columns(
        'ValueTable',
        {
            'dc:extent': 'multivalued',
            'datatype': {
                'base': 'string',
                'format': '[a-zA-Z0-9_\\-]+',
            },
            'propertyUrl': 'http://cldf.clld.org/v1.0/terms.rdf#exampleReference',
            'separator': ';',
            'name': 'Example_IDs',
        })
    cldf.add_component('ExampleTable')
    cldf.add_table(
        'constructions.csv',
        'http://cldf.clld.org/v1.0/terms.rdf#id',
        'http://cldf.clld.org/v1.0/terms.rdf#languageReference',
        'http://cldf.clld.org/v1.0/terms.rdf#name',
        'http://cldf.clld.org/v1.0/terms.rdf#description')
    cldf.add_table(
        'cvalues.csv',
        'http://cldf.clld.org/v1.0/terms.rdf#id',
        'http://cldf.clld.org/v1.0/terms.rdf#parameterReference',
        {'name': 'Construction_ID',
         'datatype': 'string',
         'required': True,
         'dc:extent': 'singlevalued'},
        'http://cldf.clld.org/v1.0/terms.rdf#codeReference',
        'http://cldf.clld.org/v1.0/terms.rdf#value')
    cldf.add_foreign_key(
        'cvalues.csv', 'Construction_ID', 'constructions.csv', 'ID')


class Dataset(BaseDataset):
    dir = pathlib.Path(__file__).parent
    id = "haspelmathindefpro"

    def cldf_specs(self):  # A dataset must declare all CLDF sets it creates.
        return CLDFSpec(
            module='StructureDataset',
            dir=self.cldf_dir,
            metadata_fname='cldf-metadata.json')

    def cmd_download(self, _args):
        """
        Download files to the raw/ directory. You can use helpers methods of `self.raw_dir`, e.g.

        >>> self.raw_dir.download(url, fname)
        """
        csv_dir = self.raw_dir / 'csv-export'
        csv_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.xlsx2csv('indefpro.xlsx', outdir=csv_dir)

    def cmd_makecldf(self, args):
        """
        Convert the raw data to a CLDF dataset.

        >>> args.writer.objects['LanguageTable'].append(...)
        """

        # read data

        csv_dir = self.raw_dir / 'csv-export'
        raw_data = list(csv_dir.read_csv('indefpro.Sheet1.csv', dicts=True))
        cparameters = make_parameters(self.etc_dir.read_csv(
            'cparameters.csv', dicts=True))
        lparameters = make_parameters(self.etc_dir.read_csv(
            'lparameters.csv', dicts=True))
        with open(self.raw_dir / 'examples.txt', encoding='utf-8') as f:
            examples = make_examples(extract_examples(f))

        # make cldf

        languages = {
            lg['ID']: lg
            for lg in make_languages(raw_data, args.glottolog.api)}
        examples = [ex for ex in examples if glosses_are_aligned(ex, languages)]
        constructions = make_constructions(raw_data)
        ccodes = make_ccodes(cparameters)
        cvalues = make_cvalues(raw_data, constructions, cparameters)
        lvalues = make_lvalues(raw_data, lparameters, examples)
        add_construction_descriptions(constructions, cvalues, cparameters)

        # write cldf

        make_schema(args.writer.cldf)

        args.writer.objects['LanguageTable'] = languages.values()
        args.writer.objects['ParameterTable'] = list(chain(
            lparameters.values(), cparameters.values()))
        args.writer.objects['ExampleTable'] = examples
        args.writer.objects['CodeTable'] = ccodes
        args.writer.objects['ValueTable'] = lvalues
        args.writer.objects['constructions.csv'] = constructions.values()
        args.writer.objects['cvalues.csv'] = cvalues
