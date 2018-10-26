# struktura datasetu
# funkce pro nacitani datasetu z nejakeho adresare
# specialni funkce pro UMAP data - extrakce jen dvojic class
# nejsou nahodou ty UMAP nagenerovana data trochu nadbytecna?
# jde to tahat promi z tech materskejch slozek 

function  getdata(dataset::String; path::String = "")
	return randn(2,4)
end