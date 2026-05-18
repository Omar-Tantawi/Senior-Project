<?php

namespace App\Http\Requests\Camera;

use Illuminate\Foundation\Http\FormRequest;

class StoreCameraRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'location'   => 'required|string|max:255',
            'isactive'   => 'sometimes|boolean',
            'code'       => 'sometimes|nullable|string|max:64|unique:camera,code',
            'stream_url' => 'sometimes|nullable|string|max:512',
        ];
    }
}
