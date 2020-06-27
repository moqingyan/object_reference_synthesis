open Grammar_tree

module Object = struct
    type color = [ `Blue | `Red | `Yellow | `Green | `Gray | `Brown | `Purple | `Cyan ]
    type size = [`Large | `Small]
    type material = [`Rubber | `Metal]
    type obj = [`Cube | `Cylinder | `Sphere | `Object1 | `Object2 | `Object | `Objects | `Target]

    type t = {
        referred : bool;
        unique : bool;
        color: color option;
        size: size option;
        material: material option;
        obj: obj;
        obj_id: int
        }

    let color_to_adj (c : color) = (c :> Grammar_tree.Adjective.t)
    let size_to_adj (c : size) = (c :> Grammar_tree.Adjective.t)
    let material_to_adj (c : material) = (c :> Grammar_tree.Adjective.t)
    let obj_to_noun (c : obj) = (c :> Grammar_tree.Noun.t)

    let maybe_color_to_maybe_adj = Option.map color_to_adj
    let maybe_size_to_maybe_adj = Option.map size_to_adj
    let maybe_material_to_maybe_adj = Option.map material_to_adj

    let to_des_np {referred; unique; color; size; material; obj; obj_id} =
        let fst : Determinant.t option =
            if (unique || referred) then Some The else
            match color, size, material, obj   with
            | Some c, _, _, _ -> Some A
            | None, Some s, _, _ -> Some A
            | None, None, Some m,_ -> Some A
            | None, None, None, o ->
                if (o = `Object1 || o = `Object2 || o = `Object || o = `Target) then Some An else Some A
        in

        let m_obj =
            match obj with
            | `Object1 -> `Object
            | `Object2 -> `Object
            | `Target -> `Object
            | a -> a
        in


        Grammar_tree.GrammarTree.NounPhrase.NP_N (
            fst,
            [maybe_color_to_maybe_adj color; maybe_size_to_maybe_adj size; maybe_material_to_maybe_adj material],
            obj_to_noun m_obj)

    (* let to_pos_np {referred; unique; color; size; material; obj} =

        Grammar_tree.GrammarTree.NounPhrase.NP_N (
            Some The,
            [],
            obj_to_noun obj) *)

    let obj_id_to_noun obj_id : Grammar_tree.Noun.t =
        match obj_id with
        | 0 -> `Target
        | 1 -> `Object1
        | _ -> `Object2

    let np_obj (obj:t) =
        print_endline("np obj" ^ string_of_int obj.obj_id);
        Grammar_tree.GrammarTree.NounPhrase.NP_N (None, [], obj_id_to_noun obj.obj_id)

    let det_np_obj (obj:t) =
        print_endline("np obj" ^ string_of_int obj.obj_id);
        Grammar_tree.GrammarTree.NounPhrase.NP_N (Some The, [], obj_id_to_noun obj.obj_id)

    let target_det_np_obj (obj : t) =
        let noun = obj_id_to_noun obj.obj_id in
        match noun with
        | `Target -> Grammar_tree.GrammarTree.NounPhrase.NP_N (Some The, [], noun)
        | _ -> Grammar_tree.GrammarTree.NounPhrase.NP_N (None, [], noun)

    let to_sentence o =
        Grammar_tree.GrammarTree.Sentence.S (
            target_det_np_obj o,
            Grammar_tree.GrammarTree.VerbPhrase.VP_N(Is, (to_des_np o))
         )

    let to_string obj =
        Grammar_tree.GrammarTree.Sentence.to_string (to_sentence obj)

end

module Relation = struct
    type rel = Left | Right | Front | Behind
    [@@derive yojson]

    type t =
    { relation : rel ;
        obj1 : Object.t ;
        obj2 : Object.t }

    let neg_rel (r: rel) : rel =
        match r with
        | Left -> Right
        | Right -> Left
        | Front -> Behind
        | Behind -> Front

    let neg {relation; obj1; obj2} =
        {relation = neg_rel relation;
         obj1 = obj2;
         obj2 = obj1}

    let to_pp {relation; obj1; obj2} =
        match relation with
        | Left ->  Grammar_tree.GrammarTree.NounPhrase.PP (
                    `To, Grammar_tree.GrammarTree.NounPhrase.NP_PP (
                         Some The, [Some (`Left :> Grammar_tree.Adjective.t)], Grammar_tree.GrammarTree.NounPhrase.PP (
                             `Of, (Object.target_det_np_obj obj2))))
        | Right ->  Grammar_tree.GrammarTree.NounPhrase.PP (
                    `To, Grammar_tree.GrammarTree.NounPhrase.NP_PP (
                         Some The, [Some (`Right :> Grammar_tree.Adjective.t)], Grammar_tree.GrammarTree.NounPhrase.PP (
                             `Of, (Object.target_det_np_obj obj2))))
        | Front -> Grammar_tree.GrammarTree.NounPhrase.PP (
                    `In, Grammar_tree.GrammarTree.NounPhrase.NP_NP_PP (
                         Grammar_tree.GrammarTree.NounPhrase.NP_N(None, [], `Front),
                         Grammar_tree.GrammarTree.NounPhrase.PP (
                             `Of, (Object.target_det_np_obj obj2))))
        | Behind -> Grammar_tree.GrammarTree.NounPhrase.PP (`Behind, Object.target_det_np_obj obj2)

    let to_sentence {relation; obj1; obj2} =
        Grammar_tree.GrammarTree.Sentence.S (
            Object.target_det_np_obj obj1,
            Grammar_tree.GrammarTree.VerbPhrase.VP_PP(Is, (to_pp {relation; obj1; obj2}))
         )

    let to_string obj =
        Grammar_tree.GrammarTree.Sentence.to_string (to_sentence obj)
end

module Combine_Rel = struct
    type rel = Left | Right | Front | Behind
    [@@derive yojson]

    type t =
    { relation : rel ;
        obj1 : Object.t ;
        obj2 : Object.t ;
        obj3 : Object.t
        }

    let neg_rel (r: rel) : rel =
        match r with
        | Left -> Right
        | Right -> Left
        | Front -> Behind
        | Behind -> Front

    let neg {relation; obj1; obj2; obj3} =
        {relation = neg_rel relation;
         obj1 = obj3;
         obj2 = obj2;
         obj3 = obj1}

    (* The objects from _ to _ are: obj1, obj2, obj3 *)
    (* Behind o1 o2 o3 = behind o1 o2  *)
    let to_sentence {relation; obj1; obj2; obj3} =
        let f =
            match relation with
            | Left -> Grammar_tree.GrammarTree.NounPhrase.PP ( `From, Grammar_tree.GrammarTree.NounPhrase.NP_JJ(`Right))
            | Right -> Grammar_tree.GrammarTree.NounPhrase.PP ( `From, Grammar_tree.GrammarTree.NounPhrase.NP_JJ(`Left ))
            | Front -> Grammar_tree.GrammarTree.NounPhrase.PP ( `From, Grammar_tree.GrammarTree.NounPhrase.NP_JJ(`Back))
            | Behind -> Grammar_tree.GrammarTree.NounPhrase.PP ( `From, Grammar_tree.GrammarTree.NounPhrase.NP_N(None, [], `Front)) in

        let t =
            match relation with
            | Left -> Grammar_tree.GrammarTree.NounPhrase.PP ( `To, Grammar_tree.GrammarTree.NounPhrase.NP_JJ(`Left))
            | Right -> Grammar_tree.GrammarTree.NounPhrase.PP ( `To, Grammar_tree.GrammarTree.NounPhrase.NP_JJ(`Right ))
            | Front ->  Grammar_tree.GrammarTree.NounPhrase.PP ( `To, Grammar_tree.GrammarTree.NounPhrase.NP_N(None, [], `Front))
            | Behind -> Grammar_tree.GrammarTree.NounPhrase.PP( `To, Grammar_tree.GrammarTree.NounPhrase.NP_JJ(`Back))   in

        let obj_des = Grammar_tree.GrammarTree.NounPhrase.NP_NPS(
                     None,
                     Object.target_det_np_obj obj1,
                    [(Punc.Comma, (Object.target_det_np_obj obj2))],
                      CC.And, Object.target_det_np_obj obj3)
        in
        Grammar_tree.GrammarTree.Sentence.S (
            Grammar_tree.GrammarTree.NounPhrase.NP_NP_PP(
                Grammar_tree.GrammarTree.NounPhrase.NP_N (None, [], `Objects),
                Grammar_tree.GrammarTree.NounPhrase.PP_PP(f, t)),
            Grammar_tree.GrammarTree.VerbPhrase.VP_N(Are, obj_des))

    let to_string obj =
        Grammar_tree.GrammarTree.Sentence.to_string (to_sentence obj)
end


module Both_Rel = struct
    type rel = Left | Right | Front | Behind
    [@@derive yojson]

    type t =
    { relation : rel ;
        obj1 : Object.t ;
        obj2 : Object.t ;
        obj3 : Object.t
        }

    (* Both obj2, obj3 are _ of obj1 *)
    let to_sentence {relation; obj1; obj2; obj3} =

        let start = Grammar_tree.GrammarTree.NounPhrase.NP_NPS ( Some Both, Object.target_det_np_obj obj2, [], And, Object.target_det_np_obj obj3 ) in
        let obj1_np = Object.target_det_np_obj obj1 in
        let pos =
            match relation with
            | Left -> Grammar_tree.GrammarTree.NounPhrase.PP ( `On,
                Grammar_tree.GrammarTree.NounPhrase.NP_NP_PP(
                    Grammar_tree.GrammarTree.NounPhrase.NP_N (Some The, [], (`Left :> Grammar_tree.Noun.t) ),
                    Grammar_tree.GrammarTree.NounPhrase.PP(`Of,  obj1_np)))
            | Right ->Grammar_tree.GrammarTree.NounPhrase.PP ( `On,
                Grammar_tree.GrammarTree.NounPhrase.NP_NP_PP(
                    Grammar_tree.GrammarTree.NounPhrase.NP_N (Some The, [], (`Right :> Grammar_tree.Noun.t) ),
                    Grammar_tree.GrammarTree.NounPhrase.PP(`Of,  obj1_np)))
            | Front ->  Grammar_tree.GrammarTree.NounPhrase.PP ( `In,
                Grammar_tree.GrammarTree.NounPhrase.NP_NP_PP(
                    Grammar_tree.GrammarTree.NounPhrase.NP_N (None, [], (`Front :> Grammar_tree.Noun.t) ),
                    Grammar_tree.GrammarTree.NounPhrase.PP(`Of,  obj1_np)))
            | Behind -> Grammar_tree.GrammarTree.NounPhrase.PP ( `Behind, obj1_np) in

        Grammar_tree.GrammarTree.Sentence.S (
            start,
            Grammar_tree.GrammarTree.VerbPhrase.VP_PP(Are, pos)
         )

    let to_string obj =
        Grammar_tree.GrammarTree.Sentence.to_string (to_sentence obj)
end

let str2color (s) =
    if (s = "blue") then
        Some `Blue
    else if (s = "yellow") then
        Some `Yellow
    else if (s = "red") then
        Some `Red
    else if (s = "green") then
        Some `Green
    else if (s = "gray") then
        Some `Gray
    else if (s = "brown") then
        Some `Brown
    else if (s = "purple") then
        Some `Purple
    else if (s = "cyan") then
        Some `Cyan
    else
        None

let str2size (s) =
    if (s = "large") then
        Some `Large
    else if (s = "small") then
        Some `Small
    else
        None

let str2mat (s) =
    if (s = "rubber") then
        Some `Rubber
    else if (s = "metal") then
        Some `Metal
    else
        None

let str2shape (s) =
    if (s = "cube") then
        `Cube
    else if (s = "cylinder") then
        `Cylinder
    else
        `Sphere
