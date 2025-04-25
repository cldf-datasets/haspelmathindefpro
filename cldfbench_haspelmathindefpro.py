import pathlib
import unicodedata
from collections import defaultdict
from itertools import chain

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


def make_languages(data, glottolog):
    gc2name = {row['Glottocode']: row['language'] for row in data}
    languoids = glottolog.languoids(ids=gc2name)
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
            'Name': 'Indefiniteness marker: {}'.format(row['form']),
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


def make_lvalues(data, lparameters):
    forms = defaultdict(list)
    for row in data:
        glottocode = row['Glottocode']
        form = row['form']
        for colname, cell in row.items():
            if (param := lparameters.get(colname)):
                forms[glottocode, param['ID']].append(form)
    return [
        {
            'ID': f'{glottocode}-{param_id}',
            'Language_ID': glottocode,
            'Parameter_ID': param_id,
            'Value': ' / '.join(param_forms),
        }
        for (glottocode, param_id), param_forms in forms.items()]


def make_schema(cldf):
    cldf.add_component('LanguageTable')
    cldf.add_component('ParameterTable')
    cldf.add_component('CodeTable')
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

    def cmd_download(self, args):
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

        # make cldf

        languages = make_languages(raw_data, args.glottolog.api)
        constructions = make_constructions(raw_data)
        ccodes = make_ccodes(cparameters)
        cvalues = make_cvalues(raw_data, constructions, cparameters)
        lvalues = make_lvalues(raw_data, lparameters)

        # write cldf

        make_schema(args.writer.cldf)

        args.writer.objects['LanguageTable'] = languages
        args.writer.objects['ParameterTable'] = list(chain(
            lparameters.values(), cparameters.values()))
        args.writer.objects['CodeTable'] = ccodes
        args.writer.objects['ValueTable'] = lvalues
        args.writer.objects['constructions.csv'] = constructions.values()
        args.writer.objects['cvalues.csv'] = cvalues
