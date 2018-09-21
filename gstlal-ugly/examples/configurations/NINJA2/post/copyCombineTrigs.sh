for i in `ls ../week*/*ALL*.sqlite`;do
	ligolw_sqlite -v --database ${i} --extract `basename ${i%%.s*}`.xml.gz
done

ligolw_add --ilwdchar-compat -v --output H1L1V1-ALL-COMBINED-871147552-876357464-LLOID.xml.gz `ls H1L1V1-ALL-8*.xml.gz`
