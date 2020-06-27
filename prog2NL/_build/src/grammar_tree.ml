module type PRINTABLE = sig
    type t

    val to_string : t -> string
end

module MakePrintOption (P : PRINTABLE) = struct
    let to_string (mv : P.t option) : string =
        match mv with
        | Some v -> P.to_string v
        | None -> ""
end

module Determinant = struct
    type t = The | An | A | Both

    let to_string d =
        match d with
        | The -> "the"
        | An -> "an"
        | A -> "a"
        | Both -> "both"
end

module Verb = struct
    type t = Is | Are

    let to_string v =
        match v with
        | Is -> "is"
        | Are -> "are"

end

module Adjective = struct
    type t =
        [ `Large | `Small
        | `Blue | `Red | `Yellow | `Green | `Gray | `Brown | `Purple | `Cyan
        | `Rubber | `Metal
        | `Right | `Left | `Back ]

    let level a : int =
        match a with
        | `Large | `Small -> 50
        | `Blue | `Red | `Yellow | `Green | `Gray | `Brown | `Purple | `Cyan -> 30
        | `Rubber | `Metal -> 10
        | `Right | `Left -> 20


    let to_string a =
        match a with
        | `Large -> "large"
        | `Small -> "small"
        | `Blue -> "blue"
        | `Red -> "red"
        | `Yellow -> "yellow"
        | `Green -> "green"
        | `Gray -> "gray"
        | `Brown -> "brown"
        | `Purple -> "purple"
        | `Cyan -> "cyan"
        | `Rubber -> "rubber"
        | `Metal -> "metal"
        | `Right -> "right"
        | `Left -> "left"
        | `Back -> "back"

end

module Noun = struct
    type t =
         [`Cube | `Cylinder | `Sphere | `Object1 | `Object2 | `Object | `Objects | `Target | `Front | `Right | `Left ]

    let to_string n =
        match n with
        | `Cube -> "cube"
        | `Cylinder -> "cylinder"
        | `Sphere -> "sphere"
        | `Object1 -> "object 1"
        | `Object2 -> "object 2"
        | `Object -> "object"
        | `Objects -> "objects"
        | `Target -> "target object"
        | `Front -> "front"
        | `Left -> "left"
        | `Right -> "right"

end

module Preposition = struct
    type t =
        [`In | `On | `Of | `Behind | `To | `From]

    let to_string p =
        match p with
        | `In -> "in"
        | `On -> "on"
        | `Of -> "of"
        | `To -> "to"
        | `Behind -> "behind"
        | `From  -> "from"

end

module Punc = struct
    type t = Comma

    let to_string c =
        match c with
        | Comma -> ","

end

module CC = struct
    type t = And

    let to_string cc =
        match cc with
        | And -> "and"

end

let join_no_empty ls = List.filter (fun x -> not (x = "")) ls |> (String.concat " ")

module GrammarTree = struct
    module DetOption = MakePrintOption (Determinant)
    module AdjOption = MakePrintOption (Adjective)
    module NounOption = MakePrintOption (Noun)
    module PrepOption = MakePrintOption (Preposition)

    module NounPhrase = struct

        type t =
            | NP_PP of Determinant.t option * Adjective.t option list * pp
            | NP_N of Determinant.t option * Adjective.t option list * Noun.t
            | NP_JJ of Adjective.t
            | NP_NP_PP of t * pp
            | NP_NP_PP_PP of t * pp * pp
            | NP_NPS of Determinant.t option * t * (Punc.t * t) list * CC.t * t
            and pp =
            | PP of Preposition.t * t
            | PP_PP of pp * pp


        let rec to_string p =
            match p with
            | NP_PP (det, adjs, pp) ->
                let detstr = DetOption.to_string det in
                let adjstr_ls = (List.map AdjOption.to_string adjs) in
                let ppstr = pp_to_string pp in
                    let detls = (if detstr == "" then [] else [detstr]) in
                    join_no_empty (detls @ adjstr_ls @ [ppstr])
            | NP_N (det, adjs, n) ->
                let detstr = DetOption.to_string det in
                let adjstr_ls = (List.map AdjOption.to_string adjs) in
                let nstr = Noun.to_string n in
                    let detls = (if detstr == "" then [] else [detstr]) in
                    join_no_empty (detls @ adjstr_ls @ [nstr])
            | NP_JJ (adj) ->
                Adjective.to_string adj
            | NP_NP_PP (np, pp) ->
                let npstr = to_string np in
                let ppstr = pp_to_string pp in
                    join_no_empty [npstr; ppstr]
            | NP_NP_PP_PP (np, pp1, pp2) ->
                let npstr = to_string np in
                let ppstr1 = pp_to_string pp1 in
                let ppstr2 = pp_to_string pp2 in
                    join_no_empty [npstr; ppstr1; ppstr2]
            | NP_NPS (det, np1, np_commas, cc, np2) ->
                let detstr = DetOption.to_string det in
                let np1_str = to_string np1 in
                let np2_str = to_string np2 in
                let cc_str = CC.to_string cc in
                let np_commas_str = join_no_empty (List.fold_left (fun ls (c, np) -> (Punc.to_string c ^ " " ^ to_string np) :: ls) [] np_commas) in
                join_no_empty [detstr; np1_str ^ np_commas_str; cc_str; np2_str]

        and pp_to_string p =
            match p with
            | PP (in_p, np) -> join_no_empty [Preposition.to_string in_p; to_string np]
            | PP_PP (pp1, pp2) -> join_no_empty [pp_to_string pp1; pp_to_string pp2]

    end

    module VerbPhrase = struct
        type t = | VP_N of Verb.t * NounPhrase.t
                 | VP_PP of Verb.t * NounPhrase.pp

        let rec to_string vp =
            match vp with
            | VP_N (verb, np) -> join_no_empty [(Verb.to_string verb); (NounPhrase.to_string np)]
            | VP_PP (verb, pp) -> join_no_empty [(Verb.to_string verb); (NounPhrase.pp_to_string pp)]

    end

    module Sentence = struct
        type t =
            | S of NounPhrase.t * VerbPhrase.t

        let to_string s =
            let S (np, vp) = s in
            let np_str = NounPhrase.to_string np in
            let vp_str = VerbPhrase.to_string vp in
            let sent_str = join_no_empty [np_str; vp_str] in
                Char.escaped (Char.uppercase_ascii (String.get sent_str 0)) ^ (String.sub sent_str 1 (String.length sent_str - 1)) ^ "."

    end

    module Reference = struct
        type t = R of Sentence.t list

        let to_string r =
            let R (sl) = r in
            let sent_sls = List.map Sentence.to_string sl in
                join_no_empty sent_sls

    end

end

open Format
open GrammarTree

(*
let a = [[The Green Object In Front Of An Object],
         [The Object Behind The Green Object]]


Reference.R (
    [
        S (The, Green, Objects, In, Front, Of, An, Object);
        S (The, Object, Behind, The, Green, Object);
    ]
)

print_string (pp_to_string p);
print_string (GrammarTree.np_to_string a);; *)