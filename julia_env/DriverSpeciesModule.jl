module DriverSpeciesModule

using Hungarian
using MatrixNetworks
using SparseArrays

export DriverSpecies, AllDriverSpecies

function DriverSpecies(AdjacencyM :: SparseMatrixCSC{Float64,Int64})
    Atemp = sparse(convert(Array, AdjacencyM))
    A = Atemp'
    SCC = scomponents(sparse(A))
    SCCs = Any[];
    for i = 1 : SCC.number
        SCCs = push!(SCCs, findall(SCC.map .== i))
    end
    nonTopLinkedSCCsIndex = Any[];
    for i = 1 : SCC.number
        minPathToSCCj = Inf
        for j = i + 1 : SCC.number
            source = SCCs[j][1]
            target = SCCs[i][1]
            d, dt, pred = bfs(sparse(A), source)
            if d[target] .== -1
                minPathFromiToj = Inf
            else
                minPathFromiToj = d[target]
            end
            if minPathFromiToj .< minPathToSCCj
               minPathToSCCj = minPathFromiToj
            end
        end
        if minPathToSCCj .== Inf
            nonTopLinkedSCCsIndex = push!(nonTopLinkedSCCsIndex,i)
        end
    end
    numberNonTopLinkedSCCs = length(nonTopLinkedSCCsIndex)
    nonTopLinkedSCCs = SCCs[nonTopLinkedSCCsIndex]
    Imatrix = zeros(size(A,1),numberNonTopLinkedSCCs)
    for i = 1 : numberNonTopLinkedSCCs
       Imatrix[nonTopLinkedSCCs[i], i] .= 1.0
    end
    Imatrix = sparse(Imatrix)
    costA = convert(Array{Union{Float64, Missing}}, abs.(sign.(A')))
    costI = convert(Array{Union{Float64, Missing}}, 2.0*abs.(sign.(Imatrix)))
    costA[findall(costA .== 0.0)] .= missing
    costI[findall(costI .== 0.0)] .= missing
    weightedBipartite = convert(Array{Union{Float64, Missing}}, hcat(costA, costI))
    assignment, cost = hungarian(weightedBipartite)
    Theta = findall( assignment .> size(A,1))
    UrMinusTheta = findall( assignment .== 0.0)
    Ur = hcat(Theta', UrMinusTheta')
    nonTopLinkedSCCsAssigned = mod.(assignment[Theta], size(A,1));
    nonTopLinkedSCCsNOTassigned = setdiff(1 : numberNonTopLinkedSCCs, nonTopLinkedSCCsAssigned)
    Au = Int64[]
    for i = 1:length(nonTopLinkedSCCsNOTassigned)
        nonTopAux = nonTopLinkedSCCs[nonTopLinkedSCCsNOTassigned[i]]
        Au = push!(Au, nonTopAux[1])
    end
    temp2 = vec(convert(Array{Int16}, Ur))
    mFDIC = unique(append!(temp2, Au))
    return(mFDIC)
end

function expand_driver_sets_from_assignment(assignment, A, nonTopLinkedSCCs; max_sets=10000)
    N = size(A, 1)
    if max_sets <= 0
        return Vector{Vector{Int64}}()
    end

    numberNonTopLinkedSCCs = length(nonTopLinkedSCCs)

    Theta = findall(assignment .> N)
    UrMinusTheta = findall(assignment .== 0.0)
    Ur = unique(vcat(Theta, UrMinusTheta))

    nonTopLinkedSCCsAssigned = mod.(assignment[Theta], N)
    nonTopLinkedSCCsNOTassigned = setdiff(1:numberNonTopLinkedSCCs, nonTopLinkedSCCsAssigned)

    choices = Vector{Vector{Int64}}()
    for scc_idx in nonTopLinkedSCCsNOTassigned
        push!(choices, collect(nonTopLinkedSCCs[scc_idx]))
    end

    sets = Vector{Vector{Int64}}()
    if isempty(choices)
        push!(sets, sort(unique(Ur)))
    else
        combo = Int64[]

        function choose(pos)
            if length(sets) >= max_sets
                return
            end
            if pos > length(choices)
                push!(sets, sort(unique(vcat(Ur, combo))))
                return
            end
            for x in choices[pos]
                push!(combo, x)
                choose(pos + 1)
                pop!(combo)
                if length(sets) >= max_sets
                    return
                end
            end
        end

        choose(1)
    end
    return sets
end

function enumerate_optimal_assignments(weightedBipartite, opt_cost, opt_count; max_assignments=10000)
    n, m = size(weightedBipartite)
    candidates = Vector{Vector{Int64}}(undef, n)

    for i in 1:n
        candidates[i] = findall(.!ismissing.(weightedBipartite[i, :]))
    end

    order = sortperm(1:n, by=i -> length(candidates[i]))

    used = falses(m)
    current = zeros(Int64, n)
    out = Vector{Vector{Int64}}()

    function backtrack(pos, assigned_count, score)
        if length(out) >= max_assignments
            return
        end
        if assigned_count > opt_count || score - opt_cost > 1e-9
            return
        end
        if pos > n
            if assigned_count == opt_count && abs(score - opt_cost) < 1e-9
                push!(out, copy(current))
            end
            return
        end

        r = order[pos]
        remaining_after = n - pos

        if assigned_count + remaining_after >= opt_count
            current[r] = 0
            backtrack(pos + 1, assigned_count, score)
        end

        if assigned_count >= opt_count
            return
        end

        for c in candidates[r]
            if !used[c]
                used[c] = true
                current[r] = c
                backtrack(pos + 1, assigned_count + 1, score + Float64(weightedBipartite[r, c]))
                used[c] = false
                current[r] = 0
            end
        end
    end

    backtrack(1, 0, 0.0)
    return out
end

function AllDriverSpecies(AdjacencyM :: SparseMatrixCSC{Float64,Int64}; max_sets=10000)
    Atemp = sparse(convert(Array, AdjacencyM))
    A = Atemp'
    SCC = scomponents(sparse(A))
    SCCs = Any[];
    for i = 1 : SCC.number
        SCCs = push!(SCCs, findall(SCC.map .== i))
    end
    nonTopLinkedSCCsIndex = Any[];
    for i = 1 : SCC.number
        minPathToSCCj = Inf
        for j = i + 1 : SCC.number
            source = SCCs[j][1]
            target = SCCs[i][1]
            d, dt, pred = bfs(sparse(A), source)
            if d[target] .== -1
                minPathFromiToj = Inf
            else
                minPathFromiToj = d[target]
            end
            if minPathFromiToj .< minPathToSCCj
               minPathToSCCj = minPathFromiToj
            end
        end
        if minPathToSCCj .== Inf
            nonTopLinkedSCCsIndex = push!(nonTopLinkedSCCsIndex,i)
        end
    end
    numberNonTopLinkedSCCs = length(nonTopLinkedSCCsIndex)
    nonTopLinkedSCCs = SCCs[nonTopLinkedSCCsIndex]
    Imatrix = zeros(size(A,1),numberNonTopLinkedSCCs)
    for i = 1 : numberNonTopLinkedSCCs
       Imatrix[nonTopLinkedSCCs[i], i] .= 1.0
    end
    Imatrix = sparse(Imatrix)
    costA = convert(Array{Union{Float64, Missing}}, abs.(sign.(A')))
    costI = convert(Array{Union{Float64, Missing}}, 2.0*abs.(sign.(Imatrix)))
    costA[findall(costA .== 0.0)] .= missing
    costI[findall(costI .== 0.0)] .= missing
    weightedBipartite = convert(Array{Union{Float64, Missing}}, hcat(costA, costI))

    assignment, opt_cost = hungarian(weightedBipartite)
    opt_count = count(assignment .> 0)
    assignments = enumerate_optimal_assignments(weightedBipartite, opt_cost, opt_count; max_assignments=max_sets)

    all_sets = Set{Tuple{Vararg{Int64}}}()
    for asg in assignments
        sets_here = expand_driver_sets_from_assignment(asg, A, nonTopLinkedSCCs; max_sets=max_sets - length(all_sets))
        for s in sets_here
            push!(all_sets, Tuple(sort(unique(s))))
        end
        if length(all_sets) >= max_sets
            break
        end
    end

    result = [collect(s) for s in all_sets]
    sort!(result, by = x -> (length(x), x))
    return result
end

end
