struct FP8
    sign::Bool             # 1 bit
    exponent::Vector{Bool} # 4 bits
    mantissa::Vector{Bool} # 3 bits
    function FP8(s::String)
        bool_vec = Bool[]
        for char in s
            if char == '1'
                push!(bool_vec, true)
            elseif char == '0'
                push!(bool_vec, false)
            elseif char == ' '
                continue
            else
                error("unexpected char $char")
            end
        end
        return new(bool_vec[1], bool_vec[2:5], bool_vec[6:8])
    end
end

function Base.show(io::IO, fp8::FP8)
    # bits
    Base.printstyled(io, "FP8 "; bold=true)
    Base.printstyled(io, fp8.sign ? 1 : 0, color=:blue)
    print(io, " ")
    exponent_str = join(map(i -> i ? "1" : "0", fp8.exponent))
    Base.printstyled(io, exponent_str, color=:red)
    print(io, " ")
    man_str = join(map(i -> i ? "1" : "0", fp8.mantissa))
    Base.printstyled(io, man_str, color=:green)
    # interpretation
    sign_str = fp8.sign ? "-1" : "+1"
    e = as_int(fp8.exponent)
    if all(fp8.exponent)
        print(io, " — sign=$sign_str, Inf/NaN")
    elseif !any(fp8.exponent)
        print(io, " — sign=$sign_str, zero")
    else
        print(io, " — sign=$sign_str, exp=2^($e-7)=2^($(e-7)), mantissa=1.$man_str")
    end
    print(io, " — decimal=$(to_decimal(fp8))")
end

# Convert an integer to a vector of Bools: the most significant
# bit comes first.
# e.g. as_bool_vec(4) = Bool[1, 0, 0]
function as_bool_vec(i::Integer)
    bool_vec = Bool[]
    while i > 0
        # should the bit be 1 or 0?
        rem = i % 2
        push!(bool_vec, rem == 1)
        # right shift
        i = div(i, 2)
    end
    return reverse(bool_vec)
end

# Convert a vector of Bools where the most significant bit comes first
# to an integer.
# e.g. as_int(Bool[1, 0, 0]) = 4
function as_int(v::Vector{Bool})
    result = 0
    pow = 0
    for bit in reverse(v)
        if bit
            result = result + (2^pow)
        end
        pow += 1
    end
    return result
end

function to_decimal(fp8::FP8)
    # sign=1 means negative
    sign = fp8.sign ? -1 : 1
    if all(fp8.exponent)
        # exponent of 1111 = Inf
        # This is not fully accurate (because of NaNs) but suffices for today
        return sign * Inf
    elseif !any(fp8.exponent)
        # exponent of 0000 = zero
        # This is also not fully accurate (because of subnormals) but suffices for today
        return sign * 0.0
    else
        pow2 = 2.0^(as_int(fp8.exponent) - 7)
        decimals = 1 + (as_int(fp8.mantissa) / 8.0)
        return sign * pow2 * decimals
    end
end

fps = [FP8("0 $exponent $mantissa")
    for exponent in ["0001", "0010", "0011"]
    for mantissa in ["000", "001", "010", "011", "100", "101", "110", "111"]
]

vals = map(to_decimal, fps)

using UnicodePlots
scatterplot(vals, ones(length(vals)), xlabel="value", height=3, width=120, marker=:circle)
