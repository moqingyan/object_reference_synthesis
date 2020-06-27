
OCAMLBUILD=ocamlbuild -use-ocamlfind
SRC=src
PKG=yojson

prog2nl:
	$(OCAMLBUILD) -I $(SRC) -pkg $(PKG) ./src/main.native
	mv main.native process_prog