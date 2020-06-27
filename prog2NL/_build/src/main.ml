open Semantic
open Map
open Yojson.Basic.Util
;;

module Int = struct
   type t = int
   (* use Pervasives compare *)
   let compare = compare
 end

module Ints = Set.Make(Int)

module Obj_mem = struct
    type t = M of (Semantic.Object.t option * Semantic.Object.t option * Semantic.Object.t option)

    let init_mem ints =
        let element_init i :Object.t  =
            let object_type =
                (match i with
                | 0 -> `Target
                | 1 -> `Object1
                | _ -> `Object2) in
                {referred = false;
                unique = false;
                color = None;
                size = None;
                material = None;
                obj = object_type;
                obj_id = i} in

        let o0 = if (Ints.exists (fun x -> x = 0) ints)
            then Some(element_init 0)
            else None in
        let o1 = if (Ints.exists (fun x -> x = 1) ints)
            then Some(element_init 1)
            else None in
        let o2 = if (Ints.exists (fun x -> x = 2) ints)
            then Some(element_init 2)
            else None in
        M(o0, o1, o2)

    let get_object (m:t) var =
        let  M(tar, v1, v2) = m in
        match var with
        | 0 -> tar
        | 1 -> v1
        | _ -> v2

    let update_object (m:t) new_element pos : t =
        let  M(tar, v1, v2) = m in
        match pos with
        | 0 -> M(new_element, v1, v2)
        | 1 -> M(tar, new_element, v2)
        | _ -> M(tar, v1, new_element)

    let maybe_o_to_sent maybe_o ls: string list =
        match maybe_o with
        | Some o -> (Object.to_string o) :: ls
        | None -> ls

    let to_sentences (m:t) : string list =
        let M(o1, o2, o3) = m in
        let ls1 = maybe_o_to_sent o1 [] in
        let ls2 = maybe_o_to_sent o2 ls1 in
        maybe_o_to_sent o3 ls2

end

module Clause = struct

    type t =
        | Attr of string * string * int
        | Pos of string * int * int
    [@@deriving yojson { exn = true }]

    let to_string c =
        match c with
        | Attr (at, av, v) -> "Attr: " ^ at ^ " " ^ av ^ " " ^ string_of_int v
        | Pos (p, v1, v2) -> "Pos: " ^ p ^ " " ^ string_of_int v1 ^ " " ^ string_of_int v2

    let to_json = to_yojson
    let from_json = of_yojson_exn

end

module MergedClause = struct
type t =
        | Attr of string * string * int
        | Pos of string * int * int
        | Comp_Pos of string * int * int * int
        | Both_Pos of string * int * int * int
    [@@deriving yojson { exn = true }]

    let to_string c =
        match c with
        | Attr (at, av, v) -> "Attr: " ^ at ^ " " ^ av ^ " " ^ string_of_int v
        | Pos (p, v1, v2) -> "Pos: " ^ p ^ " " ^ string_of_int v1 ^ " " ^ string_of_int v2
        | Comp_Pos (p, v0, v1, v2) -> "Comp pos: " ^ p ^ " " ^ string_of_int v0 ^ " " ^ string_of_int v1 ^ " " ^ string_of_int v2
        | Both_Pos (p, v0, v1, v2) -> "Both pos: " ^ p ^ " " ^ string_of_int v0 ^ " " ^ string_of_int v1 ^ " " ^ string_of_int v2

    let clause2Object (c: t) (o : Semantic.Object.t option) : Object.t option =
        match c, o with
        | Attr (attr_type, attr_val, var), None ->
            let obj_type = match var with
                    | 0 -> `Target
                    | 1 -> `Object1
                    | _ -> `Object2 in
            let new_obj:Object.t =
                {referred = false;
                unique = false;
                color = None;
                size = None;
                material = None;
                obj = obj_type;
                obj_id = var} in

            Some {new_obj with obj = obj_type}
        | Attr (attr_type, attr_val, var), Some obj ->
            if attr_type = "size" then
                    Some {obj with size = str2size (attr_val) }
                else if attr_type = "material" then
                    Some {obj with material= str2mat (attr_val)}
                else if attr_type = "shape" then
                    Some {obj with obj = str2shape (attr_val) }
                else
                    Some {obj with color = str2color (attr_val)}
        | _ -> None

    let clause2Rel (c : t) (o1: Object.t) (o2: Object.t) : Relation.t option =


        match c with
        | Pos (position, _, _) ->
            if position = "right" then
                Some {
                    relation = Relation.Left ;
                    obj1 = o1 ;
                    obj2 = o2
                }
            else if position = "left" then
                Some {
                    relation = Relation.Right ;
                    obj1 = o1 ;
                    obj2 = o2
                }
            else if position = "front" then
                Some {
                    relation = Relation.Behind ;
                    obj1 = o1 ;
                    obj2 = o2
                }
            else
                Some {
                    relation = Relation.Front ;
                    obj1 = o1 ;
                    obj2 = o2
                }
        | _ -> None

    let clause2CompRel (c : t)  (o1: Object.t) (o2: Object.t) (o3: Object.t) : Combine_Rel.t option =
        match c with
        | Comp_Pos (position, _, _, _) ->
            if position = "right" then
                Some {
                    relation = Combine_Rel.Right ;
                    obj1 = o1 ;
                    obj2 = o2 ;
                    obj3 = o3 ;
                }
            else if position = "left" then
                Some {
                    relation = Combine_Rel.Left ;
                    obj1 = o1 ;
                    obj2 = o2 ;
                    obj3 = o3 ;
                }
            else if position = "front" then
                Some {
                    relation = Combine_Rel.Front ;
                    obj1 = o1 ;
                    obj2 = o2 ;
                    obj3 = o3 ;
                }
            else
                Some {
                    relation = Combine_Rel.Behind ;
                    obj1 = o1 ;
                    obj2 = o2 ;
                    obj3 = o3 ;
                }
        | _ -> None

    let clause2BothRel (c : t)  (o1: Object.t) (o2: Object.t) (o3: Object.t) : Both_Rel.t option =
        match c with
        | Both_Pos (position, _, _, _) ->
            if position = "right" then
                Some {
                    relation = Both_Rel.Right ;
                    obj1 = o1 ;
                    obj2 = o2 ;
                    obj3 = o3 ;
                }
            else if position = "left" then
                Some {
                    relation = Both_Rel.Left ;
                    obj1 = o1 ;
                    obj2 = o2 ;
                    obj3 = o3 ;
                }
            else if position = "front" then
                Some {
                    relation = Both_Rel.Front ;
                    obj1 = o1 ;
                    obj2 = o2 ;
                    obj3 = o3 ;
                }
            else
                Some {
                    relation = Both_Rel.Behind ;
                    obj1 = o1 ;
                    obj2 = o2 ;
                    obj3 = o3 ;
                }
        | _ -> None
end

module Reference = struct
    type t = R of MergedClause.t list

    let combine_pos_clauses clause_list: t =
        let get_pos_clauses p = List.filter
            (fun c ->
            match c with
            | Clause.Pos (pc, _, _) when (pc = p) -> true
            | _ -> false )
        clause_list in
        let attr_clauses = List.filter
             (fun c ->
            match c with
            | Clause.Attr (_, _, _) -> true
            | _ -> false )
            clause_list in
        let neg r =
            match r with
            | "left" -> "right"
            | "right" -> "left"
            | "behind" -> "front"
            | _ -> "behind"
        in
        let to_merged c: MergedClause.t =
            match c with
            | Clause.Pos (r, o1, o2) -> MergedClause.Pos(r, o1, o2)
            | Clause.Attr (at, av, o) -> MergedClause.Attr (at, av, o)
        in
        let find_order ctp : MergedClause.t option =
            match ctp with
            | Clause.Pos (r, o1, o2), Clause.Pos (r1, o3, o4) when (r = r1) && o1 = o4 ->
                (** left o1 o2, left o3 o1*)
                Some (MergedClause.Comp_Pos (r, o3, o1, o2))
            | Clause.Pos (r, o1, o2), Clause.Pos (r1, o3, o4) when (r = r1) && o2 = o3 ->
                (** left o1 o2, left o2 o4*)
                Some (MergedClause.Comp_Pos (r, o1, o2, o4))
            | Clause.Pos (r, o1, o2), Clause.Pos (r1, o3, o4) when (r = r1) && o1 = o3 ->
                (** left o1 o2 *) (** left o1 o4 *)
                Some (MergedClause.Both_Pos (r, o1, o2, o4))
            | Clause.Pos (r, o1, o2), Clause.Pos (r1, o3, o4) when (r = r1) && o2 = o4 ->
                Some (MergedClause.Both_Pos (neg r, o2, o1, o3))
            | _ -> None
        in
        (* Let cls be filtered single pos clauses *)
        let rec const_merged_cls cls (ls: MergedClause.t list): MergedClause.t list=
            match cls with
            | [] -> ls
            | hd :: tl ->
                let start_with_hd = List.map (fun x -> (hd, x)) tl in
                let merged_hd_ls = List.map (fun tp -> find_order tp) start_with_hd in
                let comb_ls = List.combine start_with_hd merged_hd_ls in

                (** Process one single layer's merged value*)
                let rec process_merged_ls pcl v v_new_tl =
                    (match v with
                    | Some MergedClause.Comp_Pos _ -> v, v_new_tl
                    | _ ->
                        (match pcl with
                        | [] -> v, v_new_tl
                        | ((s, tl_v), None) :: t -> process_merged_ls t v v_new_tl
                        | ((s, tl_v), Some c) :: t ->
                            (match c with
                            | MergedClause.Comp_Pos _-> Some c, (List.filter (fun x -> x != tl_v) tl)
                            | MergedClause.Both_Pos _ -> process_merged_ls t (Some c) (List.filter (fun x -> x != tl_v) tl)
                            | _ -> process_merged_ls t v v_new_tl)))
                in
                let v, v_tl = process_merged_ls comb_ls None tl in
                match v with
                | None -> const_merged_cls tl ((to_merged hd)::ls)
                | Some c -> const_merged_cls v_tl (c :: ls)
            in
            let merged_attr = List.map to_merged attr_clauses in
            let merged_pos p =
                let pos_clauses = get_pos_clauses p in
                let merged_p = const_merged_cls (get_pos_clauses p) [] in
                print_endline (p ^ (string_of_int (List.length pos_clauses)));
                merged_p
            in
            let merged_clause_ls = merged_attr @ merged_pos "left" @ merged_pos "right"  @ merged_pos "behind" @ merged_pos "front" in
                print_endline (string_of_int (List.length merged_clause_ls));
                R (merged_clause_ls)

    let rec get_mentioned_rec (cls_ls: MergedClause.t list) (cur: Ints.t): Ints.t =
        match cls_ls with
        | [] -> cur
        | hd :: tl  ->
            match hd with
            | MergedClause.Attr (attr_type, attr_val, var) ->
                (get_mentioned_rec tl (Ints.add var cur))
            | MergedClause.Pos (position, var1, var2) ->
                let cur1 = Ints.add var1 cur in
                (get_mentioned_rec tl (Ints.add var2 cur1))
            | MergedClause.Both_Pos (_, var1, var2, var3) ->
                let new_set = List.fold_left (fun set x -> Ints.add x set) cur [var1; var2; var3] in
                (get_mentioned_rec tl new_set)
            | MergedClause.Comp_Pos  (_, var1, var2, var3) ->
                let new_set = List.fold_left (fun set x -> Ints.add x set) cur [var1; var2; var3] in
                (get_mentioned_rec tl new_set)

    let get_mentioned (r:t) : Ints.t =
        let R (cls_ls) = r in
        get_mentioned_rec cls_ls Ints.empty

end

let construct_obj_mem_single (m:Obj_mem.t) (hd:MergedClause.t) : Obj_mem.t =
    match hd with
        | MergedClause.Attr (attr_type, attr_val, var) ->
            let new_obj = MergedClause.clause2Object hd (Obj_mem.get_object m var) in
            let new_mem_tp = Obj_mem.update_object m new_obj var in
            new_mem_tp
        | _ -> m

let construct_obj_mem  (cls_ls: Reference.t): Obj_mem.t =
    let mentioned = Reference.get_mentioned cls_ls in
    let mem = Obj_mem.init_mem mentioned in
    let Reference.R (ls) = cls_ls in
    List.fold_left construct_obj_mem_single mem ls

let construct_rel_sentences (m: Obj_mem.t) (cls_ls: Reference.t) : string list =

    let construct_rel_sentence_single (sentences: string list) (hd: MergedClause.t) : string list =
         match hd with
        | MergedClause.Attr (attr_type, attr_val, var) ->
            sentences
        | MergedClause.Pos (position, var1, var2) ->
            print_endline ("Pos");
            let maybe_o1 = Obj_mem.get_object m var1 in
            let maybe_o2 = Obj_mem.get_object m var2 in
            (match maybe_o1, maybe_o2 with
                | Some o1, Some o2 ->
                    let maybe_rel = MergedClause.clause2Rel hd o1 o2 in
                    (match maybe_rel with
                    | Some rel -> (Semantic.Relation.to_string rel) :: sentences
                    | None -> sentences)
                | _ ->
                print_endline ("not found: " ^ string_of_int var1 ^ " or " ^ string_of_int var2);
                sentences)
        | MergedClause.Comp_Pos (position, var1, var2, var3) ->
            print_endline ("Comp Pos");
            let maybe_o1 = Obj_mem.get_object m var1 in
            let maybe_o2 = Obj_mem.get_object m var2 in
            let maybe_o3 = Obj_mem.get_object m var3 in
            (match maybe_o1, maybe_o2, maybe_o3 with
                | Some o1, Some o2, Some o3 ->
                    let maybe_rel = MergedClause.clause2CompRel hd o1 o2 o3 in
                    (match maybe_rel with
                    | Some rel -> (Semantic.Combine_Rel.to_string rel) :: sentences
                    | None -> sentences)
                | _ ->
                print_endline ("not found: " ^ string_of_int var1 ^ " or " ^ string_of_int var2 ^ " or " ^ string_of_int var3);
                sentences)
        | MergedClause.Both_Pos (position, var1, var2, var3) ->
            print_endline ("Both Pos");
            let maybe_o1 = Obj_mem.get_object m var1 in
            let maybe_o2 = Obj_mem.get_object m var2 in
            let maybe_o3 = Obj_mem.get_object m var3 in
            (match maybe_o1, maybe_o2, maybe_o3 with
                | Some o1, Some o2, Some o3 ->
                    let maybe_rel = MergedClause.clause2BothRel hd o1 o2 o3 in
                    (match maybe_rel with
                    | Some rel -> (Semantic.Both_Rel.to_string rel) :: sentences
                    | None -> sentences)
                | _ ->
                print_endline ("not found: " ^ string_of_int var1 ^ " or " ^ string_of_int var2 ^ " or " ^ string_of_int var3);
                sentences)
        in
    let Reference.R ls = cls_ls in
        List.fold_left construct_rel_sentence_single [] ls

let print_obj o =
    match o with
    | Some x -> "s"
    | None -> "n"

let construct_sentences (cls_ls: Reference.t) : string list =
    (* Initialize the mem *)
    let mem = construct_obj_mem cls_ls in
    (* Generate relations *)
    let obj_sentences = Obj_mem.to_sentences mem in
    let ref_sentences = construct_rel_sentences mem cls_ls in
    List.append obj_sentences ref_sentences;;


let program_json = Yojson.Basic.from_file Sys.argv.(1) in
let oc = open_out Sys.argv.(2) in
let prog_list = (program_json |> Yojson.Basic.Util.to_list ) in
let const_cls p =
    let ps = (p |> Yojson.Basic.Util.filter_string) in
    let pi = (p |> Yojson.Basic.Util.filter_int) in

    (match List.length ps with
    | 1 -> Clause.Pos ( List.hd ps, List.hd pi, List.hd (List.tl pi))
    | _ -> Clause.Attr (List.hd ps, List.hd (List.tl ps), List.hd pi))
    in
let to_clause_list p = (p |> Yojson.Basic.Util.to_list) in
let clause_list = List.map to_clause_list prog_list in
    print_endline("blah");
let cls = List.map const_cls clause_list in
    print_endline("cls0");
    print_endline (string_of_int (List.length cls));
    let merged_cls = Reference.combine_pos_clauses cls in
    print_endline("cls1");
    let sentences = construct_sentences merged_cls in
    print_endline("cls2");
    (* List.map print_endline sentences;; *)
    output_string oc (String.concat " " sentences);;

