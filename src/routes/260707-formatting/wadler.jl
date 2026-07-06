abstract type FExpr end
struct Identifier <: FExpr
    name::String
end
struct FuncCall <: FExpr
    name::String
    args::Vector{FExpr}
end

const EX1 = FuncCall("myfunc", [Identifier("x"), Identifier("y")])

const EX2 = FuncCall(
    "myfunc",
    [Identifier("x"), FuncCall("myfunc", [Identifier("y"), Identifier("z")])],
)

abstract type Doc end
struct NilD <: Doc end
struct TextD <: Doc
    text::String
    doc::Doc
end
struct LineD <: Doc
    indent::Int
    ws::Int
    doc::Doc
end
struct UnionD <: Doc
    doc1::Doc
    doc2::Doc
end

function show_doc(io::IO, ::NilD, depth=0)
    println(io, " " ^ depth * "NilD")
end
function show_doc(io::IO, d::TextD, depth=0)
    println(io, " " ^ depth * "TextD $(repr(d.text))")
    show_doc(io, d.doc, depth + 2)
end
function show_doc(io::IO, d::LineD, depth=0)
    println(io, " " ^ depth * "LineD $(d.indent) $(d.ws)")
    show_doc(io, d.doc, depth + 2)
end
function show_doc(io::IO, d::UnionD, depth=0)
    println(io, " " ^ depth * "UnionD")
    show_doc(io, d.doc1, depth + 2)
    show_doc(io, d.doc2, depth + 2)
end
Base.show(io::IO, d::Doc) = show_doc(io, d)

flatten(::NilD) = NilD()
flatten(d::TextD) = TextD(d.text, flatten(d.doc))
# Convert linebreak to whitespace
flatten(d::LineD) = TextD(" " ^ d.ws, flatten(d.doc))
# By definition, `d.doc1` and `d.doc2` must flatten to the same thing
flatten(d::UnionD) = flatten(d.doc1)

group(::NilD) = NilD()
group(d::TextD) = TextD(d.text, group(d.doc))
group(d::LineD) = UnionD(flatten(d), d)
group(d::UnionD) = UnionD(group(d.doc1), d.doc2)

pprint(::NilD) = ""
pprint(d::TextD) = d.text * pprint(d.doc)
pprint(d::LineD) = "\n" * (" " ^ d.indent) * pprint(d.doc)
pprint(::UnionD) = error("pprint UnionD")

INDENT = 4
to_doc(a::Identifier) = TextD(a.name, NilD())
function to_doc(a::FuncCall)
    is_last = true
    arg_doc = TextD(")", NilD())
    for arg in reverse(a.args)
        indent = is_last ? 0 : INDENT
        tail = group(LineD(indent, is_last ? 0 : 1, arg_doc))
        arg_and_comma = "$(arg.name)$(is_last ? "" : ",")"
        arg_doc = TextD(arg_and_comma, tail)
        is_last = false
    end
    TextD("$(a.name)(", group(LineD(INDENT, 0, arg_doc)))
end

best(::Int, ::Int, ::NilD) = NilD()
function best(width::Int, col::Int, d::TextD)
    TextD(d.text, best(width, col + textwidth(d.text), d.doc))
end
function best(width::Int, ::Int, d::LineD)
    LineD(d.indent, best(width, d.indent, d.doc))
end
function best(width::Int, col::Int, d::UnionD)
    best1 = best(width, col, d.doc1)
    best2 = best(width, col, d.doc2)
    better(width, col, best1, best2)
end
function better(width::Int, col::Int, d1::Doc, d2::Doc)
    fits(width - col, d1) ? d1 : d2
end

fits(remaining_width::Int, ::NilD) = remaining_width >= 0
fits(remaining_width::Int, d::TextD) =
    remaining_width >= 0 && fits(remaining_width - textwidth(d.text), d.doc)
fits(remaining_width::Int, ::LineD) = remaining_width >= 0
