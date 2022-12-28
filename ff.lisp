;; feed forward networks based on the book
;; Rashid (2016) Make your own neural network, CreateSpace Independent Publishing Platform.

;; our first representation of a network can be a sequence of 2d arrays
;; so we need to be able to make an array of any two-dimensions and then
;; initialize it with random entries

;; matrix multiplication
;; to-do: throw an error if the dimensions don't match, but cs2 and rs1 should be equal
(defun matmul (a2 a1)
  (destructuring-bind (rs2 cs2) (array-dimensions a2)
    (destructuring-bind (rs1 cs1) (array-dimensions a1)
      (when (not (= cs2 rs1)) (error "bad array dimensions in multiplication"))
      (let ((new-array (make-array (list rs2 cs1))))
        (dotimes (r2 rs2)
          (dotimes (c1 cs1)
            (let ((sum 0))
              (dotimes (r1 rs1)
                (incf sum (* (aref a2 r2 r1) (aref a1 r1 c1))))
              (setf (aref new-array r2 c1) sum))))
        new-array))))

(defun matmul-flip (a1 a2)
  (matmul a2 a1))

(defun transpose (a)
  (destructuring-bind (rs cs) (array-dimensions a)
    (let ((new-array (make-array (list cs rs))))
      (dotimes (r rs)
        (dotimes (c cs)
          (setf (aref new-array c r) (aref a r c))))
      new-array)))

(defun zeros (inodes onodes)
  (make-array (list onodes inodes) :initial-element 0))

;; specifically 2d version
(defun map-array-2d (fn a)
  (destructuring-bind (rs cs) (array-dimensions a)
    (let ((new-array (make-array (list rs cs) :initial-element 0)))
      (dotimes (r rs)
        (dotimes (c cs)
          (setf (aref new-array r c) (funcall fn (aref a r c)))))
      new-array)))
        
;; this is the general version that uses the total array size and row major order
(defun map-array (fn a)
  (let ((s (array-total-size a))
        (new-array (make-array (array-dimensions a))))
    (dotimes (i s)
      (setf (row-major-aref new-array i) (funcall fn (row-major-aref a i))))
    new-array))

(defun random-array (inodes onodes)
  (map-array #'(lambda (x) (random 1.0)) (zeros inodes onodes)))

;; if we really want to make this work correctly we should play with which type-specifier is used and how the ordering is done 
(defun zip (s1 s2)
  (map (type-of s1) #'(lambda (x y) (list x y)) s1 s2))

(defun connection-sizes (ls)
  (zip ls (cdr ls)))

(defun make-ff-net (layers)
  (mapcar #'(lambda (p) (apply #'random-array p)) (connection-sizes layers)))

(defun run-network (nn i)
  (reduce #'(lambda (a1 a2) (map-array #'sigmoid (matmul a2 a1))) nn :initial-value i))

(defun sigmoid (x)
  (/ 1.0 (+ 1.0 (exp (- x)))))

(defun promote-vec (v)
  (make-array (list 1 (length v)) :initial-contents (list v)))

(defun make-vector (&optional (initsize 10))
  (make-array initsize :adjustable t :fill-pointer 0))

;; I really want to write a macro for looping through arrays of arbitrary dimension with all their row and column names

(defun demote-array (a)
  (destructuring-bind (r c) (array-dimensions a)
    (when (not (or (= r 1) (= c 1))) (error "can't demote 2d array"))
    (cond ((= 1 r c) (aref a 0 0))
          ((= r 1) (let ((new-vec (make-vector)))
                     (dotimes (i c)
                       (vector-push (aref a 0 i) new-vec))
                     new-vec))
          ((= c 1) (let ((new-vec (make-vector)))
                     (dotimes (i r)
                       (vector-push (aref a i 0) new-vec))
                     new-vec)))))
        
(defun dot (v1 v2)
  (reduce #'(lambda (n l) (+ n (* (car l) (cadr l)))) (zip v1 v2) :initial-value 0))

;; here a neural network is a 
(defun train (nn input target)
  (let* ((output (run-network nn input))
         (err (- target output)))
    (
